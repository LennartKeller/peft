import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from peft import FXLoraConfig, LoraConfig, XLoraConfig, get_peft_model
from peft.peft_model import get_model_status


# from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING


# print(*PEFT_TYPE_TO_TUNER_MAPPING.items(), sep="\n")

target = os.getenv("TARGET", "fxlora").lower()
if target == "fxlora":
    print("Using FXLoraConfig")
    target_cfg_cls = FXLoraConfig
    kwargs = {"num_active": 2}
else:
    print("Using XLoraConfig")
    target_cfg_cls = XLoraConfig
    kwargs = {}


def get_model(model_name_or_path, device="cpu"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_cache=False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = model.config
    return model, tokenizer, config


def noisy_lora_b(model, verbose: bool = True):
    """
    Adds Gaussian noise to lora_B matrices
    to ensure that gradients can flow into
    the adapters on first forward/backward pass.
    """
    hits = 0
    for name, param in model.named_parameters():
        if "lora_B" in name:
            hits += 1
            param.data.normal_(mean=0.0, std=0.005)
    if verbose:
        print("Total lora_B layers modified with noise:", hits)
    return model


def check_classifier_gradients(model):
    base_model_obj = model.get_base_model()
    if hasattr(base_model_obj, "internal_xlora_classifier"):
        classifier = base_model_obj.internal_xlora_classifier
        if hasattr(classifier, "_last_hidden_state"):
            hs = classifier._last_hidden_state
            if hs.grad is not None:
                print(f"Hidden state grad sum: {hs.grad.abs().sum().item():.6f}")
                print(f"Hidden state grad max: {hs.grad.abs().max().item():.6f}")
            else:
                print("Hidden state grad is None!")


def check_all_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad and "classifier" in name:
            if param.grad is not None:
                grad_sum = param.grad.abs().sum().item()
                grad_max = param.grad.abs().max().item()
                print(f"{name} --> sum={grad_sum:.6f}, max={grad_max:.6f}")
            else:
                print(f"{name} --> None")


if __name__ == "__main__":
    print("=== START: Preparing dummy adapters ===")
    # model_name_or_path = "distilbert/distilbert-base-uncased"
    # def get_model(model_name_or_path):
    #     model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    #     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    #     config = model.config
    #     return model, tokenizer, config

    model_name_or_path = "openai-community/gpt2"

    # Create Lora Adapters
    if torch.backends.mps.is_available():
        print("Using MPS device for adapter creation")
        device = "mps:0"
    elif torch.cuda.is_available():
        print("Using CUDA device for adapter creation")
        device = "cuda:0"
    else:
        print("Using CPU device for adapter creation")
        device = "cpu"

    model, tokenizer, config = get_model(model_name_or_path, device=device)
    adapters_configs = {
        "0": LoraConfig(r=8, lora_alpha=32, target_modules="all-linear"),
        "1": LoraConfig(r=4, lora_alpha=16, target_modules="all-linear"),
        "2": LoraConfig(r=2, lora_alpha=8, target_modules="all-linear"),
        "3": LoraConfig(r=1, lora_alpha=4, target_modules="all-linear"),
        "4": LoraConfig(r=1, lora_alpha=2, target_modules="all-linear"),
    }
    adapter_paths = {}
    for adapter_name, adapter_config in adapters_configs.items():
        adapter_path = f"debug/{model_name_or_path.split('/')[-1]}/{adapter_name}"
        if Path(adapter_path).exists():
            print(f"Adapter path {adapter_path} already exists, skipping creation.")
            adapter_paths[adapter_name] = adapter_path
            adapter_model = "dummy"
            continue
        adapter_model = get_peft_model(model, adapter_config)
        adapter_model.save_pretrained(adapter_path)
        adapter_paths[adapter_name] = adapter_path
    del model, tokenizer, config, adapter_model
    print("=== END: Preparing dummy adapters ===")

    print("=== START: Loading PEFT model with multiple adapters ===")
    model, tokenizer, config = get_model(model_name_or_path)
    print(model)
    print(target_cfg_cls)
    lora_config = target_cfg_cls(
        task_type="CAUSAL_LM", hidden_size=config.hidden_size, xlora_depth=4, adapters=adapter_paths, **kwargs
    )
    model = get_peft_model(model, lora_config)
    print(model)
    print("PEFT_STATUS:", get_model_status(model))
    model.print_trainable_parameters()
    print("=== END: Loading PEFT model with multiple adapters ===")

    model.enable_adapter_layers()
    model.train()

    model = noisy_lora_b(model)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    labels = inputs["input_ids"].clone()
    inputs["labels"] = labels

    print("=== START: Testing 1. Forward and backward pass with adapters (0, 1) ===")
    model.activate_adapters(["0", "1"], offload_device="cpu")
    print("PEFT-STATUS after activating adapters:", get_model_status(model))
    outputs = model(**inputs.to(model.device))
    loss = outputs.loss
    print("Loss:", loss.item())
    loss.backward(retain_graph=True)
    print("=== END: Testing 1. Forward and backward pass ===")

    print("=== START: Testing 2. Forward and backward pass with adapters (0, 2) ===")
    model.activate_adapters(["0", "2"], offload_device="cpu")
    print("PEFT-STATUS after activating adapters:", get_model_status(model))
    outputs = model(**inputs.to(model.device))
    loss = outputs.loss
    print("Loss:", loss.item())
    loss.backward(retain_graph=True)
    print("=== END: Testing 2. Forward and backward pass ===")

    print("=== START: Testing 3. Forward and backward pass with adapters (1, 2) ===")
    model.activate_adapters(["1", "2"], offload_device="cpu")
    print("PEFT-STATUS after activating adapters:", get_model_status(model))
    outputs = model(**inputs.to(model.device))
    loss = outputs.loss
    print("Loss:", loss.item())
    loss.backward(retain_graph=True)
    print("=== END: Testing 3. Forward and backward pass ===")

    check_all_gradients(model)
    check_classifier_gradients(model)
    print("=== ALL TESTS COMPLETED SUCCESSFULLY ===")

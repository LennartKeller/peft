import os


try:
    from rich import print

    def print_dbg(*args, **kwargs):
        if DEBUG:
            print("[bold red][PEFT DEBUG][/bold red]", *args, **kwargs)
except ImportError:

    def print_dbg(*args, **kwargs):
        if DEBUG:
            print("[PEFT DEBUG]", *args, **kwargs)


DEBUG = bool(int(os.getenv("DEBUG", 0)))

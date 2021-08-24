import traceback


class BlockingContext:
    def __init__(self, save_stack=True):
        print("__init__")
        self.save_stack_ = save_stack
        "\n".join(traceback.format_stack(limit=5))

    def __enter__(self):
        print("__enter__")
        return self.save_stack_

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_stack_

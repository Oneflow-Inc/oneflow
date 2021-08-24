import traceback
import oneflow._oneflow_internal


class BlockingContext:
    def __init__(self, save_stack=True):
        print("__init__")
        self.save_stack_ = save_stack
        stack_info = "\n".join(traceback.format_stack(limit=5))
        oneflow._oneflow_internal.blocking.register_stack_info_callback(
            lambda: stack_info
        )

    def __enter__(self):
        print("__enter__")
        return self.save_stack_

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_stack_

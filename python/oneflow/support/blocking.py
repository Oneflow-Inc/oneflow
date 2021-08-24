import traceback
import oneflow._oneflow_internal


class BlockingContext:
    def __init__(self, save_stack=True):
        self.save_stack_ = save_stack
        stack_info = "\n".join(traceback.format_stack(limit=5))
        oneflow._oneflow_internal.blocking.register_stack_info_callback(
            lambda: stack_info
        )

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
        # oneflow._oneflow_internal.blocking.clear_stack_info_callback()

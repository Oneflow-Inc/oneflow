import oneflow as flow

_ONEFLOW_EXEC_MODE = False


class oneflow_exec_mode(object):
    def __init__(self, enabled=None):
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = True

    def __enter__(self):
        global _ONEFLOW_EXEC_MODE
        self.prev_mode = _ONEFLOW_EXEC_MODE
        _ONEFLOW_EXEC_MODE = self.enabled
        self.prev_grad_mode = flow.is_grad_enabled()
        _ = flow.set_grad_enabled(False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _ONEFLOW_EXEC_MODE
        _ONEFLOW_EXEC_MODE = self.prev_mode
        _ = flow.set_grad_enabled(self.prev_grad_mode)


def oneflow_exec_mode_enabled():
    global _ONEFLOW_EXEC_MODE
    return _ONEFLOW_EXEC_MODE

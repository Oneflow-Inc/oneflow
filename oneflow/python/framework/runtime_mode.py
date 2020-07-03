from contextlib import contextmanager

NORMAL_MODE = "NORMAL_MODE"
GLOBAL_MODE = "GLOBAL_MODE"
DEVICE_MODE = "DEVICE_MODE"


def CurrentMode():
    return mode_statck[0]


def IsValidMode(mode):
    return mode == NORMAL_MODE or mode == GLOBAL_MODE or mode == DEVICE_MODE


@contextmanager
def ModeScope(mode):
    global mode_statck
    mode_statck.insert(0, mode)
    yield
    mode_statck.pop(0)


mode_statck = [NORMAL_MODE]

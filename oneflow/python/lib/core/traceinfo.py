import os
import traceback


def GetFrameLocationStr(depth=-1):
    assert depth < 0
    frame = traceback.extract_stack()[depth - 1]
    return "%s:%d" % (frame[0], frame[1])


def GetStackInfoExcludeOneflowPythonFile():
    import oneflow

    dirname = os.path.dirname(oneflow.__file__)
    stack_info = traceback.extract_stack()
    filtered_stack_info = filter(
        lambda x: x[0].startswith(dirname) == False, stack_info
    )
    return list(filtered_stack_info)

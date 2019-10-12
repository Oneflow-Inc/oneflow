import traceback
import os


def GetStackInfoExcludeOneflowPythonFile():
    import oneflow

    dirname = os.path.dirname(oneflow.__file__)
    stack_info = traceback.extract_stack()
    filtered_stack_info = filter(
        lambda x: x[0].startswith(dirname) == False, stack_info
    )
    return list(filtered_stack_info)

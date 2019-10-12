import traceback

def GetStackInfo():
    try:
        raise _ThisIsNotAnException
    except _ThisIsNotAnException:
        return traceback.format_stack()[0:-1]

class _ThisIsNotAnException(Exception):
    def __init__(self):
        pass

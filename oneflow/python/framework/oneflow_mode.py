def IsCurrentCompileMode:
    return current_mode == COMPILE_MODE
    
def IsCurrentRuntimeMode:
    return current_mode == RUNTIME_MODE

COMPILE_MODE = 'COMPILE_MODE';
RUNTIME_MODE = 'RUNTIME_MODE';

current_mode = None;

class CompileMode:
    def __init__(self):
        assert current_mode == None, "no reentrant use of oneflow_mode"

    def __enter__(self):
        current_mode = COMPILE_MODE

    def __exit__(self, *args):
        current_mode = None
        
class RuntimeMode:
    def __init__(self):
        assert current_mode == None, "no reentrant use of oneflow_mode"

    def __enter__(self):
        current_mode = RUNTIME_MODE

    def __exit__(self, *args):
        current_mode = None


def remote(func):
    TODO()

def static_assert(func):
    TODO()

def main(func):
    def Main(*arg):
        if oneflow_mode.IsCurrentCompileMode():
            TODO()
        else:
            func(*arg)
    return main;

def config(func):
    TODO()

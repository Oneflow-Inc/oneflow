import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.decorator_context as decorator_context

def remote(func):
    TODO()

def static_assert(func):
    TODO()

def main(func):
    def Main(*arg):
        if oneflow_mode.IsCurrentCompileMode():
            if hasattr(func, '__config_func__'):
                func.__config_func__(compile_context.cur_job_set)
        else:
            func(*arg)
    decorator_context.main_func = Main
    return Main;

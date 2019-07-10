from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.decorator_context as decorator_context
import oneflow.python.framework.oneflow_mode as oneflow_mode
import oneflow.python.framework.compiler as compiler
import oneflow.python.framework.runtime as runtime

def remote(func):
    def Func(*arg):
        if oneflow_mode.IsCurrentCompileMode():
            return compiler.CompileJob(func)
        elif oneflow_mode.IsCurrentRuntimeMode():
            return runtime.LaunchJob(func.__name__, *arg)
        else:
            raise NotImplementedError
    Func.__name__ = func.__name__
    decorator_context.job_name2func[Func.__name__] = Func
    if hasattr(func, '__config_func__'):
        Main.__config_func__ = func.__config_func__
    else:
        def EmptyConfig(job_conf):
            pass
        Main.__config_func__ = EmptyConfig
    return Func

def static_assert(func):
    TODO()

def main(func):
    def Main(*arg):
        func(*arg)
    decorator_context.main_func = Main
    if hasattr(func, '__config_func__'):
        Main.__config_func__ = func.__config_func__
    else:
        def EmptyConfig(job_set):
            pass
        Main.__config_func__ = EmptyConfig
    return Main;

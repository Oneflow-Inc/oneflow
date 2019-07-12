from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.decorator_context as decorator_context
import oneflow.python.framework.oneflow_mode as oneflow_mode
import oneflow.python.framework.runtime as runtime
import oneflow.python.framework.val as val
import oneflow.python.framework.var as var
import oneflow.python.lib.core.func_inspect_util as func_inspect_util

def remote(func):
    def Func(*arg):
        if oneflow_mode.IsCurrentCompileMode():
            return func(*arg)
        elif oneflow_mode.IsCurrentRuntimeMode():
            return runtime.LaunchJob(func.__name__, *arg)
        else:
            raise NotImplementedError
    Func.__name__ = func.__name__
    for x in dir(func):
        if x.startswith('__oneflow_'):
            setattr(Func, x, getattr(func, x))
    if hasattr(Func, '__oneflow_arg_default__') == False:
        Func.__oneflow_arg_default__ = AssertAndGetArgDefaults(func)
    decorator_context.job_name2func[Func.__name__] = Func
    if hasattr(Func, '__oneflow_config_func__') == False:
        Func.__oneflow_config_func__ = lambda x: None
    return Func

def static_assert(func):
    TODO()

def main(func):
    def Main(*arg):
        func(*arg)
    decorator_context.main_func = Main
    if hasattr(func, '__oneflow_config_func__'):
        Main.__oneflow_config_func__ = func.__oneflow_config_func__
    else:
        def EmptyConfig(job_set):
            pass
        Main.__oneflow_config_func__ = EmptyConfig
    return Main;

def AssertAndGetArgDefaults(func):
    for arg_name, arg_default_val in func_inspect_util.GetArgNameAndDefaultTuple(func):
        assert isinstance(arg_default_val, val.val) or isinstance(arg_default_val, var.var)
    return func_inspect_util.GetArgDefaults(func)
    

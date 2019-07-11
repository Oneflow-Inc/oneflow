from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.decorator_context as decorator_context
import oneflow.python.framework.oneflow_mode as oneflow_mode

def compose_config(*decorators):
    assert len(decorators) > 0
    ret_decorator = decorators[-1]
    decorators = decorators[0:-1][::-1]
    for decorator in decorators:
        ret_decorator = _ComposeConfig(decorator, ret_decorator)
    return ret_decorator

def MakeResourceConfigDecorator(field, field_type):
    return _MakeConfigDecorator(_AssertIsCompilingMain, _UpdateMainDecoratorContext,
                                lambda job_set: job_set.resource, field, field_type)
def MakeIOConfigDecorator(field, field_type):
    return _MakeConfigDecorator(_AssertIsCompilingMain, _UpdateMainDecoratorContext,
                                lambda job_set: job_set.io_conf, field, field_type)
def MakeCppFlagsConfigDecorator(field, field_type):
    return _MakeConfigDecorator(_AssertIsCompilingMain, _UpdateMainDecoratorContext,
                                lambda job_set: job_set.cpp_flags_conf, field, field_type)
def MakeProfilerConfigDecorator(field, field_type):
    return _MakeConfigDecorator(_AssertIsCompilingMain, _UpdateMainDecoratorContext,
                                lambda job_set: job_set.profile_conf, field, field_type)

def MakeJobOtherConfigDecorator(field, field_type):
    return _MakeConfigDecorator(_AssertIsCompilingRemote, _UpdateRemoteDecoratorContext,
                                lambda job_conf: job_conf.other, field, field_type)

def DefaultConfigJobSet(job_set):
    assert compile_context.IsCompilingMain()
    _DefaultConfigResource(job_set)
    _DefaultConfigIO(job_set)
    _DefaultConfigCppFlags(job_set)

def DefaultConfigJobConf(job_conf):
    assert oneflow_mode.IsCurrentCompileMode()
    assert compile_context.IsCompilingMain() == False

def _ComposeConfig(first_decorator, second_decorator):
    def Decorator(func):
        return first_decorator(second_decorator(func))
    return Decorator

def _AssertIsCompilingMain():
    assert oneflow_mode.IsCurrentCompileMode(), \
        "config decorators are merely allowed to use when compile"
    assert compile_context.IsCompilingMain()
    
def _AssertIsCompilingRemote():
    assert oneflow_mode.IsCurrentCompileMode(), \
        "config decorators are merely allowed to use when compile"
    assert compile_context.IsCompilingRemote()
    
def _ComposeConfigFunc(config_func, other_config_func):
    def ComposedConfigFunc(job_set_or_job_conf):
        config_func(job_set_or_job_conf)
        other_config_func(job_set_or_job_conf)
    return ComposedConfigFunc

def _UpdateDecorateConfigFunc(decorated_func, config_func, func):
    if hasattr(func, '__oneflow_config_func__'):
        decorated_func.__oneflow_config_func__ = \
            _ComposeConfigFunc(config_func, func.__oneflow_config_func__)
    else:
        decorated_func.__oneflow_config_func__ = config_func

def _UpdateMainDecoratorContext(decorated_func, func):
    if decorator_context.main_func == func:
        decorator_context.main_func = decorated_func
    else:
        assert decorator_context.main_func is None, "no mutltiply 'main' decorator supported"

def _UpdateRemoteDecoratorContext(decorated_func, func):
    job_name = func.__name__
    if decorator_context.job_name2func.get(job_name) == func:
        decorator_context.job_name2func = decorated_func
    else:
        assert job_name not in decorator_context.job_name2func, \
            "no mutltiply 'remote' decorator supported"

def _GenConfigDecorator(config_func, decorator_ctx_handler):
    def Decorator(func):
        def DecoratedFunc(*argv):
            func(*argv)
            
        DecoratedFunc.__name__ = func.__name__
        decorator_ctx_handler(DecoratedFunc, func)
        for x in dir(func):
            if x.startswith('__oneflow_'):
                setattr(DecoratedFunc, x, getattr(func, x))
        _UpdateDecorateConfigFunc(DecoratedFunc, config_func, func)
        return DecoratedFunc
    
    return Decorator

def _MakeConfigDecorator(ctx_asserter,
                         decorator_ctx_handler,
                         get_attr_container,
                         field, field_type):
    def Func(val):
        assert isinstance(val, field_type), \
            "config field '%s' should be instance of %s, %s given"%(field, field_type, type(val))
        def ConfigFunc(pb_msg):
            ctx_asserter()
            setattr(get_attr_container(pb_msg), field, val)
        return _GenConfigDecorator(ConfigFunc, decorator_ctx_handler)
    return Func

def _DefaultConfigResource(job_set):
    resource = job_set.resource
    if len(resource.machine) == 0:
        machine = resource.machine.add()
        machine.id = 0
        machine.addr = "127.0.0.1"
    if resource.HasField("ctrl_port") == False:
        resource.ctrl_port = 2017
    if resource.HasField("gpu_device_num") == False:
        resource.gpu_device_num = 1
    if resource.HasField("cpu_device_num") == False:
        resource.cpu_device_num = 1

def _DefaultConfigIO(job_set):
    io_conf = job_set.io_conf
    if io_conf.data_fs_conf.WhichOneof("fs_type") == None:
        io_conf.data_fs_conf.localfs_conf.SetInParent()
    if io_conf.snapshot_fs_conf.WhichOneof("fs_type") == None:
        io_conf.snapshot_fs_conf.localfs_conf.SetInParent()
        
def  _DefaultConfigCppFlags(job_set):
    job_set.cpp_flags_conf.SetInParent()

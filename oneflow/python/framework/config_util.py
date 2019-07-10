from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.decorator_context as decorator_context
import oneflow.python.framework.oneflow_mode as oneflow_mode


def MakeResourceConfigDecorator(field, field_type):
    return _MakeConfigDecorator(lambda job_set: job_set.resource, field, field_type)
def MakeIOConfigDecorator(field, field_type):
    return _MakeConfigDecorator(lambda job_set: job_set.io_conf, field, field_type)
def MakeCppFlagsConfigDecorator(field, field_type):
    return _MakeConfigDecorator(lambda job_set: job_set.cpp_flags_conf, field, field_type)
def MakeProfilerConfigDecorator(field, field_type):
    return _MakeConfigDecorator(lambda job_set: job_set.profile_conf, field, field_type)

def DefaultConfigJobSet(job_set):
    assert compile_context.IsCompilingMain()
    _DefaultConfigResource(job_set)
    _DefaultConfigIO(job_set)
    _DefaultConfigCppFlags(job_set)

def _GenConfigFunc(config_func, other_config_func):
    def composed_config_func(job_set_or_job_conf):
        assert oneflow_mode.IsCurrentCompileMode(), \
            "config decorators are merely allowed to use when compile"
        assert compile_context.IsCompilingMain()
        config_func(job_set_or_job_conf)
        other_config_func(job_set_or_job_conf)
    return composed_config_func

def _UpdateDecorateFuncAndContext(decorated_func, config_func, func):
    if hasattr(func, '__config_func__'):
        decorated_func.__config_func__ = _GenConfigFunc(config_func, func.__config_func__)
    else:
        decorated_func.__config_func__ = config_func
    if decorator_context.main_func == func:
        decorator_context.main_func = decorated_func
    else:
        assert decorator_context.main_func is None, "only single main func supported"

def _GenConfigDecorator(config_func):
    def decorator(func):
        def decorated_func(*argv):
            func(*argv)

        _UpdateDecorateFuncAndContext(decorated_func, config_func, func)
        decorated_func.__name__ = func.__name__
        return decorated_func
    
    return decorator

def _MakeConfigDecorator(get_attr_container, field, field_type):
    def Func(val):
        assert isinstance(val, field_type), \
            "config field '%s' should be instance of %s, %s given"%(field, field_type, type(val))
        def ConfigFunc(pb_msg):
            setattr(get_attr_container(pb_msg), field, val)
        return _GenConfigDecorator(ConfigFunc)
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

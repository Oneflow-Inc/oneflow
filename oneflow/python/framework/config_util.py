from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.decorator_context as decorator_context
import oneflow.python.framework.decorator_util as decorator_util
import oneflow.python.framework.placement_context as placement_context
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.oneflow_mode as oneflow_mode
import oneflow.core.job.resource_pb2 as resource_util

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
def machine(machines):
    def ConfigFunc(job_conf):
        _AssertIsCompilingMain()
        job_conf.resource.machine.extend(_MakeMachine(machines))
    return _GenConfigDecorator(ConfigFunc, _UpdateMainDecoratorContext)

def MakeJobOtherConfigDecorator(field, field_type):
    return _MakeConfigDecorator(_AssertIsCompilingRemote, _UpdateRemoteDecoratorContext,
                                lambda job_conf: job_conf.other, field, field_type)
def config_train_by_func(cb):
    def ConfigFunc(job_conf):
        _AssertIsCompilingRemote()
        job_conf.other.predict_conf.tmp_split_fw_bw_train_conf.SetInParent()
        cb(job_conf.other.predict_conf.tmp_split_fw_bw_train_conf)
    return _GenConfigDecorator(ConfigFunc, _UpdateRemoteDecoratorContext)

def placement(device_names):
    def ConfigFunc(job_conf):
        _AssertIsCompilingRemote()
        job_conf.other.predict_conf.tmp_split_fw_bw_train_conf.SetInParent()
        cb(job_conf.other.predict_conf.tmp_split_fw_bw_train_conf)
    parallel_conf = placement_util.MakeParallelConf(device_names)
    def UpdateContext(decorated_func, func):
        _UpdateRemoteDecoratorContext(decorated_func, func)
        placement_context.job_name2default_parallel_conf[func.__name__] = parallel_conf
    placement_scope = placement_util.PlacementScope(parallel_conf)
    decorator = _GenConfigDecorator(ConfigFunc, UpdateContext)
    placement_scope.__call__ = lambda self, func: decorator(func)
    return placement_scope

def DefaultConfigJobSet(job_set):
    assert compile_context.IsCompilingMain()
    _DefaultConfigResource(job_set)
    _DefaultConfigIO(job_set)
    _DefaultConfigCppFlags(job_set)

def DefaultConfigJobConf(job_conf):
    assert oneflow_mode.IsCurrentCompileMode()
    assert compile_context.IsCompilingMain() == False
    _DefaultConfigJobConf(job_conf)

def _MakeMachine(machines):
    if isinstance(machines, str): machines = [machines]
    resource = resource_util.Resource()
    rp_machine = resource.machine
    for m_data in machines:
        m = rp_machine.add()
        if isinstance(m_data, str):
            m.addr = m_data
        elif isinstance(m_data, dict):
            if 'addr' in m_data: m.addr = m_data['addr']
            if 'ctrl_port_agent' in m_data: m.ctrl_port_agent = m_data['ctrl_port_agent']
            if 'data_port_agent' in m_data: m.data_port_agent = m_data['data_port_agent']
        else:
            raise NotImplementedError
    id = 0
    addrs_for_check = set()
    for m in rp_machine:
        m.id = id
        id += 1
        assert m.addr not in addrs_for_check
        addrs_for_check.add(m.addr)
    return rp_machine
    
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
    if hasattr(func, '__oneflow_arg_default__') == False:
        func.__oneflow_arg_default__ = decorator_util.AssertAndGetArgDefaults(func)
    decorated_func.__oneflow_arg_default__ = func.__oneflow_arg_default__

def _UpdateMainDecoratorContext(decorated_func, func):
    if decorator_context.main_func == func:
        decorator_context.main_func = decorated_func
    else:
        assert decorator_context.main_func is None, "no mutltiply 'main' decorator supported"

def _UpdateRemoteDecoratorContext(decorated_func, func):
    job_name = func.__name__
    if job_name in decorator_context.job_name2func and \
            decorator_context.job_name2func[job_name] == func:
        decorator_context.job_name2func[job_name] = decorated_func
    else:
        assert job_name not in decorator_context.job_name2func, \
            "no mutltiply 'remote' decorator supported"

def _GenConfigDecorator(config_func, decorator_ctx_handler):
    def Decorator(func):
        def DecoratedFunc(*argv):
            return func(*argv)
            
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

def _DefaultConfigIO(job_set):
    io_conf = job_set.io_conf
    if io_conf.data_fs_conf.WhichOneof("fs_type") == None:
        io_conf.data_fs_conf.localfs_conf.SetInParent()
    if io_conf.snapshot_fs_conf.WhichOneof("fs_type") == None:
        io_conf.snapshot_fs_conf.localfs_conf.SetInParent()
        
def  _DefaultConfigCppFlags(job_set):
    job_set.cpp_flags_conf.SetInParent()

    
def _DefaultConfigJobConf(job_conf):
    assert job_conf.other.HasField('piece_size'), "batch_size unset"
    other = job_conf.other
    if other.WhichOneof("job_type") is None:
        other.predict_conf.SetInParent()
    if other.HasField('train_conf'):
        other.train_conf.batch_size = other.piece_size
    if other.predict_conf.HasField('tmp_split_fw_bw_train_conf'):
        other.predict_conf.tmp_split_fw_bw_train_conf.batch_size = other.piece_size

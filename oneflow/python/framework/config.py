from __future__ import absolute_import

import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.decorator_context as decorator_context
import oneflow.python.framework.oneflow_mode as oneflow_mode

def config_resource(machine = None,
                    ctrl_port = None,
                    data_port = None,
                    gpu_device_num = None,
                    cpu_device_num = None,
                    comm_net_worker_num = None,
                    max_mdsave_worker_num = None,
                    use_rdma = None,
                    rdma_mem_block_mbyte = None,
                    rdma_recv_msg_buf_mbyte = None,
                    reserved_host_mem_mbyte = None,
                    reserved_device_mem_mbyte = None):
    def config_func(job_set):
        resource = job_set.resource
        if machine is not None:
            resource.machine = machine
        if ctrl_port is not None:
            resource.ctrl_port = ctrl_port
        if data_port is not None:
            resource.data_port = data_port
        if gpu_device_num is not None:
            resource.gpu_device_num = gpu_device_num
        if cpu_device_num is not None:
            resource.cpu_device_num = cpu_device_num
        if comm_net_worker_num is not None:
            resource.comm_net_worker_num = comm_net_worker_num
        if max_mdsave_worker_num is not None:
            resource.max_mdsave_worker_num = max_mdsave_worker_num
        if use_rdma is not None:
            resource.use_rdma = use_rdma
        if rdma_mem_block_mbyte is not None:
            resource.rdma_mem_block_mbyte = rdma_mem_block_mbyte
        if rdma_recv_msg_buf_mbyte is not None:
            resource.rdma_recv_msg_buf_mbyte = rdma_recv_msg_buf_mbyte
        if reserved_host_mem_mbyte is not None:
            resource.reserved_host_mem_mbyte = reserved_host_mem_mbyte
        if reserved_device_mem_mbyte is not None:
            resource.reserved_device_mem_mbyte = reserved_device_mem_mbyte
        
    return _GenJobSetConfigDecorator(config_func)

def config_io(data_fs_conf = None,
              snapshot_fs_conf = None,
              model_load_snapshot_path = None,
              model_save_snapshots_path = None,
              enable_write_snapshot = None,
              save_downloaded_file_to_local_fs = None,
              persistence_buf_byte = None):
    def config_func(job_set):
        io_conf = job_set.io_conf
        if data_fs_conf is not None:
            io_conf.data_fs_conf = data_fs_conf
        if snapshot_fs_conf is not None:
            io_conf.snapshot_fs_conf = snapshot_fs_conf
        if model_load_snapshot_path is not None:
            io_conf.model_load_snapshot_path = model_load_snapshot_path
        if model_save_snapshots_path is not None:
            io_conf.model_save_snapshots_path = model_save_snapshots_path
        if enable_write_snapshot is not None:
            io_conf.enable_write_snapshot = enable_write_snapshot
        if save_downloaded_file_to_local_fs is not None:
            io_conf.save_downloaded_file_to_local_fs = save_downloaded_file_to_local_fs
        if persistence_buf_byte is not None:
            io_conf.persistence_buf_byte = persistence_buf_byte
        
    return _GenJobSetConfigDecorator(config_func)
    
def config_cpp_flags(log_dir = None,
                     logtostderr = None,
                     logbuflevel = None,
                     v = None,
                     grpc_use_no_signal = None):
    def config_func(job_set):
        cpp_flags_conf = job_set.cpp_flags_conf
        if log_dir is not None:
            cpp_flags_conf.log_dir = log_dir
        if logtostderr is not None:
            cpp_flags_conf.logtostderr = logtostderr
        if logbuflevel is not None:
            cpp_flags_conf.logbuflevel = logbuflevel
        if v is not None:
            cpp_flags_conf.v = v
        if grpc_use_no_signal is not None:
            cpp_flags_conf.grpc_use_no_signal = grpc_use_no_signal
        
    return _GenJobSetConfigDecorator(config_func)
 
def config_profiler(collect_act_event = None):
    def config_func(job_set):
        profile_conf = job_set.profile_conf
        if collect_act_event is not None:
            profile_conf.collect_act_event = collect_act_event
        
    return _GenJobSetConfigDecorator(config_func)

def DefaultConfigJobSet(job_set):
    assert compile_context.IsCompilingMain()
    _DefaultConfigResource(job_set)
    _DefaultConfigIO(job_set)
    _DefaultConfigCppFlags(job_set)

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

def _GenJobSetConfigDecorator(config_func):
    def decorator(func):
        def decorated_func(*argv):
            func(*argv)

        _UpdateDecorateFuncAndContext(decorated_func, config_func, func)
        return decorated_func
    
    return decorator

def _UpdateDecorateFuncAndContext(decorated_func, config_func, func):
    if hasattr(func, '__config__func__'):
        decorated_func.__config__func__ = _GenConfigFunc(config_func, func.__config__func__)
    else:
        decorated_func.__config__func__ = config_func
    if decorator_context.main_func == func:
        decorator_context.main_func = decorated_func
    else:
        assert decorator_context.main_func is None, "only single main func supported"

def _GenConfigFunc(config_func, other_config_func):
    def composed_config_func(job_set):
        assert oneflow_mode.IsCurrentCompileMode(), \
            "config decorators are merely allowed to use when compile"
        assert compile_context.IsCompilingMain()
        config_func(job_set)
        other_config_func(job_set)
    return composed_config_func

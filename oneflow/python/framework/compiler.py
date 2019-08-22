from __future__ import absolute_import

import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.python.lib.core.func_inspect_util as func_inspect_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.input_blob_def as input_blob_def
import oneflow.python.ops as ops
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('get_cur_job_conf_builder')
def get_cur_job_conf_builder():
    return config_util.JobConfigProtoBuilder(compile_context.cur_job.job_conf)

def Compile(job_set, job_func):
    check_unique_job_func_name = set()
    job = job_set.job.add()
    compile_context.ResetCurJob(job)
    _CompileJob(job, job_func, config_util.inited_config_proto)
    config_util.TryCompleteDefaultJobConfigProto(job.job_conf)
    assert job.job_conf.job_name not in check_unique_job_func_name
    check_unique_job_func_name.add(job.job_conf.job_name)
    compile_context.ResetCurJob(None)

def _CompileJob(job, func, config):
    job_name = func.__name__
    device_type, machine_dev_ids = placement_util.GetDefaultMachineDeviceIds(config.resource)
    func.__oneflow_input_remote_blobs__ = []
    interface_op_names = []
    with placement_util.DevicePriorPlacementScope(device_type, machine_dev_ids):
        for blob_desc in _GetArgDefault(func):
            assert isinstance(blob_desc, input_blob_def.input_blob_def)
            remote_input_blob = ops.InputOpByBlobDesc(blob_desc)
            func.__oneflow_input_remote_blobs__.append(remote_input_blob)
            interface_op_names.append(remote_input_blob.op_name)
        ret_remote_blobs = func(*func.__oneflow_input_remote_blobs__)
        if ret_remote_blobs is None:
            func.__oneflow_output_remote_blobs__ = None 
        elif isinstance(ret_remote_blobs, remote_blob_util.RemoteBlob):
            func.__oneflow_output_remote_blobs__ = ops.RetOpByRemoteBlob(ret_remote_blobs)
            interface_op_names.append(func.__oneflow_output_remote_blobs__.op_name)
        elif isinstance(ret_remote_blobs, tuple) or isinstance(ret_remote_blobs, list):
            func.__oneflow_output_remote_blobs__ = []
            for remote_blob in ret_remote_blobs:
                assert isinstance(remote_blob, remote_blob_util.RemoteBlob)
                output_remote_blob = ops.RetOpByRemoteBlob(remote_blob)
                func.__oneflow_output_remote_blobs__.append(output_remote_blob)
                interface_op_names.append(output_remote_blob.op_name)
            if isinstance(ret_remote_blobs, tuple):
                func.__oneflow_output_remote_blobs__ = tuple(func.__oneflow_output_remote_blobs__)
        else:
            raise NotImplementedError
    job.job_conf.job_name = job_name
    job.job_conf.arg_op_name.extend(interface_op_names)

def _GetArgDefault(func):
    if hasattr(func, '__oneflow_arg_default__'): return func.__oneflow_arg_default__
    return func_inspect_util.GetArgDefaults(func)

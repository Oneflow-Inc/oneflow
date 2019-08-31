from __future__ import absolute_import

import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.python.lib.core.func_inspect_util as func_inspect_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.input_blob_def as input_blob_def
import oneflow.python.framework.job_builder as job_builder
import oneflow.python.ops as ops
from oneflow.python.oneflow_export import oneflow_export

@oneflow_export('get_cur_job_conf_builder')
def get_cur_job_conf_builder():
    return config_util.JobConfigProtoBuilder(compile_context.cur_job.job_conf)

def Compile(job_set, job_func):
    check_unique_job_func_name = set()
    job = job_set.job.add()
    compile_context.ResetCurJob(job)
    job.job_conf.job_name = job_func.__name__
    with job_builder.JobBuildAndInferCtx(job.job_conf.job_name):
        _CompileJob(job_func, config_util.inited_config_proto)
        job_builder.CurCtxSetJobConfIfNotSet(job.job_conf)
        assert job_builder.CurCtxHasJobConf()
    config_util.TryCompleteDefaultJobConfigProto(job.job_conf)
    assert job.job_conf.job_name not in check_unique_job_func_name
    check_unique_job_func_name.add(job.job_conf.job_name)
    compile_context.ResetCurJob(None)

def _CompileJob(func, config):
    device_type, machine_dev_ids = placement_util.GetDefaultMachineDeviceIds(config.resource)
    func.__oneflow_input_blob_defs__ = _GetArgDefault(func)
    with placement_util.DevicePriorPlacementScope(device_type, machine_dev_ids):
        ret_remote_blobs = _CompileJobBody(func)
        if ret_remote_blobs is None:
            func.__oneflow_output_remote_blobs__ = None 
        elif isinstance(ret_remote_blobs, remote_blob_util.RemoteBlob):
            func.__oneflow_output_remote_blobs__ = ops.RetOpByRemoteBlob(ret_remote_blobs)
        elif isinstance(ret_remote_blobs, tuple) or isinstance(ret_remote_blobs, list):
            func.__oneflow_output_remote_blobs__ = []
            for remote_blob in ret_remote_blobs:
                assert isinstance(remote_blob, remote_blob_util.RemoteBlob)
                output_remote_blob = ops.RetOpByRemoteBlob(remote_blob)
                func.__oneflow_output_remote_blobs__.append(output_remote_blob)
            if isinstance(ret_remote_blobs, tuple):
                func.__oneflow_output_remote_blobs__ = tuple(func.__oneflow_output_remote_blobs__)
        else:
            raise NotImplementedError

def _CompileJobBody(func):
    input_op_add_and_infered = False
    def AddAndInferInputOp():
        if input_op_add_and_infered: return
        for blob_desc in func.__oneflow_input_blob_defs__:
            assert isinstance(blob_desc, input_blob_def.input_blob_def)
            ops.InputOpByBlobDesc(blob_desc)
        input_op_add_and_infered = True
    with compile_context.BeforeNonInputOpBuildAndInferHook(AddAndInferInputOp):
        ret_remote_blobs = func()
    if input_op_add_and_infered == False: AddAndInferInputOp()
    return ret_remote_blobs


def _GetArgDefault(func):
    if hasattr(func, '__oneflow_arg_default__'): return func.__oneflow_arg_default__
    return func_inspect_util.GetArgDefaults(func)

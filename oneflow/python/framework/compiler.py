from __future__ import absolute_import

import oneflow.core.job.job_pb2 as job_util
import oneflow.python.lib.core.func_inspect_util as func_inspect_util
import oneflow.python.lib.core.pb_util as pb_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.input_blob_def as input_blob_def
import oneflow.python.ops as ops
from oneflow.python.lib.core.box import Box

from contextlib import contextmanager

from oneflow.python.oneflow_export import oneflow_export
import oneflow.python.framework.c_api_util as c_api_util

def Compile(function_desc, config_proto):
    job_conf = function_desc.job_config_proto
    job_conf.job_name = function_desc.job_func.__name__
    placement_scope = function_desc.function_attribute.default_placement_scope
    if placement_scope is None:
        dev_ids = placement_util.GetDefaultMachineDeviceIds(config_proto.resource)
        placement_scope = placement_util.DevicePriorPlacementScope(*dev_ids)

    compile_context.ResetCurJobContext()
    with _JobBuildAndInferCtx(job_conf.job_name), placement_scope:
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_conf)
        _CompileJob(function_desc.job_func)

def _CompileJob(func):
    func.__oneflow_input_blob_defs__ = _GetArgDefault(func)
    inputs = _AddAndInferInputOps(func)
    func.__oneflow_output_remote_blobs__ = _RecursiveMakeRetRemoteBlobs(func(*inputs))

def _RecursiveMakeRetRemoteBlobs(out_remote_blobs):
    if out_remote_blobs is None: return None
    if isinstance(out_remote_blobs, (input_blob_def.input_blob_def, remote_blob_util.RemoteBlob)):
        return ops.RetOpByRemoteBlob(out_remote_blobs)
    if isinstance(out_remote_blobs, (tuple, list)):
        return type(out_remote_blobs)(_RecursiveMakeRetRemoteBlobs(x) for x in out_remote_blobs)
    if isinstance(out_remote_blobs, dict):
        return {k : _RecursiveMakeRetRemoteBlobs(v) for k, v in out_remote_blobs.items()}
    raise NotImplementedError

def _AddAndInferInputOps(func):
    return [ops.InputOpByBlobDesc(blob_desc) for blob_desc in func.__oneflow_input_blob_defs__]

def _GetArgDefault(func):
    if hasattr(func, '__oneflow_arg_default__'): return func.__oneflow_arg_default__
    return func_inspect_util.GetArgDefaults(func)

@contextmanager
def _JobBuildAndInferCtx(job_name):
    c_api_util.JobBuildAndInferCtx_Open(job_name)
    yield
    c_api_util.JobBuildAndInferCtx_Close()


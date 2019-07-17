from __future__ import absolute_import

import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.python.lib.core.func_inspect_util as func_inspect_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.val as val
import oneflow.python.framework.ops as ops

def get_cur_job_conf_builder():
    return config_util.JobConfigProtoBuilder(compile_context.cur_job.job_conf)

def Compile(job_set, job_funcs):
    config_util.TryCompleteDefaultConfigProto(job_set.config)
    check_unique_job_func_name = set()
    for job_func in job_funcs:
        job = job_set.job.add()
        compile_context.cur_job = job
        _CompileJob(job, job_func, job_set.config)
        config_util.TryCompleteDefaultJobConfigProto(job.job_conf)
        assert job.job_conf.job_name not in check_unique_job_func_name
        check_unique_job_func_name.add(job.job_conf.job_name)
        compile_context.cur_job = None

def _CompileJob(job, func, config):
    job_name = func.__name__
    parallel_conf = placement_util.GetJobPlacementParallelConf(job_name, config.resource)
    compile_context.job_name2input_remote_blobs[job_name] = []
    input_remote_blobs = compile_context.job_name2input_remote_blobs[job_name]
    ret_remote_blobs = None
    interface_op_names = []
    with placement_util.PlacementScope(parallel_conf):
        for blob_desc in _GetArgDefault(func):
            assert isinstance(blob_desc, val.val)
            remote_input_blob = ops.InputOpByBlobDesc(blob_desc)
            input_remote_blobs.append(remote_input_blob)
            interface_op_names.append(remote_input_blob.op_name)
        ret_remote_blobs = func(*input_remote_blobs)
        if ret_remote_blobs is None: ret_remote_blobs = ()
        if isinstance(ret_remote_blobs, remote_blob_util.RemoteBlob):
            ret_remote_blobs = (ret_remote_blobs,)
        assert isinstance(ret_remote_blobs, tuple) or isinstance(ret_remote_blobs, list)
        compile_context.job_name2output_remote_blobs[job_name] = []
        output_remote_blobs = compile_context.job_name2output_remote_blobs[job_name]
        for remote_blob in ret_remote_blobs:
            assert isinstance(remote_blob, remote_blob_util.RemoteBlob)
            output_remote_blob = ops.OutputOpByRemoteBlob(remote_blob)
            output_remote_blobs.append(output_remote_blob)
            interface_op_names.append(output_remote_blob.op_name)
    job.job_conf.job_name = job_name
    job.job_conf.arg_op_name.extend(interface_op_names)

def _GetArgDefault(func):
    if hasattr(func, '__oneflow_arg_default__'): return func.__oneflow_arg_default__
    return func_inspect_util.GetArgDefaults(func)

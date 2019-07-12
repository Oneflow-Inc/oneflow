from __future__ import absolute_import

import oneflow.core.job.job_conf_pb2 as job_conf_util
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.python.framework.decorator_context as decorator_context
import oneflow.python.framework.oneflow_mode as oneflow_mode
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.placement_util as placement_util
import oneflow.python.framework.config_util as config_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.val as val
import oneflow.python.framework.var as var
import oneflow.python.framework.ops as ops

def Compile():
    assert oneflow_mode.IsCurrentCompileMode(), "Compile() must be under compile mode"
    assert decorator_context.main_func is not None, "no main function found"
    assert len(decorator_context.job_name2func) > 0, "no job function found"
    compile_context.cur_job_set = job_set_util.JobSet()
    with compile_context.CompilingMain():
        job_set = compile_context.cur_job_set
        decorator_context.main_func.__oneflow_config_func__(job_set)
        config_util.DefaultConfigJobSet(job_set)
    with compile_context.CompilingRemote():
        for job_name, func in decorator_context.job_name2func.items():
            compile_context.cur_job = job_conf_util.JobConf()
            func.__oneflow_config_func__(compile_context.cur_job)
            config_util.DefaultConfigJobConf(compile_context.cur_job)
            compile_context.cur_job.job_name = func.__name__
            CompileJob(func)
            job_set.job_conf.add().CopyFrom(compile_context.cur_job)
    return job_set

def GetMainFunc():
    return decorator_context.main_func

def CompileJob(func):
    job_name = func.__name__
    parallel_conf = placement_util.GetJobPlacementParallelConf(
        job_name, compile_context.cur_job_set.resource)
    assert hasattr(func, '__oneflow_arg_default__')
    compile_context.job_name2input_remote_blobs[job_name] = []
    input_remote_blobs = compile_context.job_name2input_remote_blobs[job_name]
    ret_remote_blobs = None
    with placement_util.PlacementScope(parallel_conf):
        for blob_desc in func.__oneflow_arg_default__:
            if isinstance(blob_desc, val.val):
                remote_input_blob = ops.InputOpByBlobDesc(blob_desc)
            elif isinstance(blob_desc, var.var):
                remote_input_blob = ops.VariableOpByBlobDesc(blob_desc)
            else:
                raise NotImplementedError
            input_remote_blobs.append(remote_input_blob)
        ret_remote_blobs = func(*input_remote_blobs)
        if ret_remote_blobs is None: ret_remote_blobs = ()
        if isinstance(ret_remote_blobs, remote_blob_util.RemoteBlob):
            ret_remote_blobs = (ret_remote_blobs,)
        assert isinstance(ret_remote_blobs, tuple) or isinstance(ret_remote_blobs, list)
        compile_context.job_name2output_remote_blobs[job_name] = []
        output_remote_blobs = compile_context.job_name2output_remote_blobs[job_name]
        for remote_blob in ret_remote_blobs:
            assert isinstance(remote_blob, remote_blob_util.RemoteBlob)
            output_remote_blobs.append(ops.OutputOpByRemoteBlob(remote_blob))

from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_context as runtime_ctx
import oneflow.python.framework.job_instance as job_instance
import oneflow.python.framework.inter_user_job_util as inter_user_job_util
import oneflow.python.framework.out_remote_blobs_result as out_remote_blobs_result
import numpy as np

def GetMachineRuntimeEnv(job_set):
    if len(job_set.config.resource.machine) == 1:
        return MasterRuntimeEnv(job_set)
    else:
        TODO()

class MasterRuntimeEnv(object):
    def __init__(self, job_set):
        self.job_set_ = job_set

    def __enter__(self):
        assert len(self.job_set_.job) > 0, "no job in job_set found"
        c_api_util.InitGlobalOneflowByJobSet(self.job_set_)
        runtime_ctx.Init()

    def __exit__(self, *args):
        runtime_ctx.Destroy()
        c_api_util.DestroyGlobalOneflow()

class WorkerRuntimeEnv(object):
    def __init__(self, job_set):
        pass

    def __enter__(self):
        TODO()
        raise ThisIsNotAnError

    def __exit__(self, *args):
        TODO()

class ThisIsNotAnError(Exception):
    pass

def LaunchJob(job_func, RunCallback, *arg):
    job_name = job_func.__name__
    assert len(arg) == len(job_func.__oneflow_input_remote_blobs__)
    for i in range(len(arg)):
        assert isinstance(arg[i], np.ndarray)
        input_op_name = job_func.__oneflow_input_remote_blobs__[i].op_name
        inter_user_job_util.AsyncPush(input_op_name, inter_user_job_util.MakePushCallback(arg[i]))
    c_api_util.LaunchJob(job_instance.MakeUserJobInstance(job_name, RunCallback))
    return out_remote_blobs_result.OutRemoteBlobsResult(job_func.__oneflow_output_remote_blobs__)

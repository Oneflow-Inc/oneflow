from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_context as runtime_ctx
import oneflow.python.framework.job_instance as job_instance
import numpy as np

def GetMachineRuntimeEnv(job_set):
    if len(job_set.resource.machine) == 1:
        return MasterRuntimeEnv(job_set)
    else:
        TODO()

class MasterRuntimeEnv:
    def __init__(self, job_set):
        self.job_set_ = job_set
        runtime_ctx.Init()

    def __enter__(self):
        assert len(self.job_set_.job_conf) > 0, "no job in job_set found"
        c_api_util.InitGlobalOneflowByJobSet(self.job_set_)
        
    def __exit__(self, *args):
        c_api_util.DestroyGlobalOneflow()

class WorkerRuntimeEnv():
    def __init__(self, job_set):
        pass

    def __enter__(self):
        TODO()
        raise ThisIsNotAnError

    def __exit__(self, *args):
        TODO()

class ThisIsNotAnError(Exception):
    pass

class LaunchJob(job_name, *arg):
    input_op_names = runtime_ctx.job_name2input_op_names[job_name]
    assert len(arg) == len(input_op_names)     
    for i in range(len(arg)):
        assert isinstance(arg[i], np.ndarray)
        push_job_name = TODO()
        op_name = input_op_names[i]
        push_cb = MakePushCallback(arg[i])
        c_api_util.LaunchJob(job_instance.MakePushJobInstance(push_job_name, op_name, push_cb))
    c_api_util.LaunchJob(job_instance.MakeUserJobInstance(job_name))
    return runtime_ctx.job_name2output_op_names[job_name]

def MakePushCallback(ndarray):
    return lambda ofblob: ofblob.CopyFromNdarray(ndarray)

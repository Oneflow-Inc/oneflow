from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.runtime_context as runtime_ctx
import oneflow.python.framework.job_instance as job_instance
import oneflow.python.framework.inter_user_job as inter_user_job
import numpy as np

def GetMachineRuntimeEnv(job_set):
    if len(job_set.resource.machine) == 1:
        return MasterRuntimeEnv(job_set)
    else:
        TODO()

class MasterRuntimeEnv(object):
    def __init__(self, job_set):
        self.job_set_ = job_set

    def __enter__(self):
        from google.protobuf import text_format
        print (text_format.MessageToString(self.job_set_))
        assert len(self.job_set_.job_conf) > 0, "no job in job_set found"
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

def LaunchJob(job_name, *arg):
    input_op_names = runtime_ctx.job_name2input_op_names[job_name]
    assert len(arg) == len(input_op_names)
    for i in range(len(arg)):
        assert isinstance(arg[i], np.ndarray)
        inter_user_job.AsyncPush(input_op_names[i], inter_user_job.MakePushCallback(arg[i]))
    c_api_util.LaunchJob(job_instance.MakeUserJobInstance(job_name))
    return runtime_ctx.job_name2output_op_names[job_name]

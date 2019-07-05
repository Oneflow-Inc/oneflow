from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util

def GetMachineRuntimeEnv(job_set):
    if job_set.resource.machine.size() == 1:
        return MasterRuntimeEnv(job_set)
    else:
        TODO()

class MasterRuntimeEnv:
    def __init__(self, job_set):
        self.job_set_ = job_set

    def __enter__(self):
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

class ThisIsNotAnError(Exception):
    pass

from __future__ import absolute_import

import oneflow.python.framework.c_api_util as c_api_util

class JobBuildAndInferCtx(object):
    def __init__(self, job_name):
        self.job_name_ = job_name

    def __enter__(self):
        _Open(self.job_name_)

    def __exit__(self, *args):
        _Close(self.job_name_)

def GetCurCtxJobName():
    return c_api_util.JobBuildAndInferCtx_GetCurrentJobName()

def CurCtxCheckJob():
    c_api_util.CurJobBuildAndInferCtx_CheckJob()

def CurCtxSetJobConfIfNotSet(job_config_proto):
    global job_conf_inited
    if job_conf_inited == False:
        c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_config_proto)
        job_conf_inited = True

def CurCtxAddAndInferOp(op_conf_proto, parallel_conf_proto):
    return c_api_util.CurJobBuildAndInferCtx_AddAndInferOp(op_conf_proto, parallel_conf_proto)

def CurCtxAddLossLogicalBlobName(lbn):
    return c_api_util.CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn)

def CurCtxAddLbiAndDiffWatcherUuidPair(lbi_and_uuid):
    return c_api_util.CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid)

def CurCtxHasJobConf():
    return c_api_util.CurJobBuildAndInferCtx_HasJobConf()

def GetStaticShape(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_GetStaticShape(job_name, lbn)

def GetDataType(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_GetDataType(job_name, lbn)

def GetBatchAxis(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_GetBatchAxis(job_name, lbn)

def GetSplitAxisFromProducerView(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn)

def GetParallelConfFromProducerView(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_GetParallelConfFromProducerView(job_name, lbn)

def _Open(job_name):
    return c_api_util.JobBuildAndInferCtx_Open(job_name)

def _Close(job_name):
    global job_conf_inited
    job_conf_inited = False
    return c_api_util.JobBuildAndInferCtx_Close()

job_conf_inited = False

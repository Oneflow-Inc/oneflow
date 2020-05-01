# global function context
import oneflow.python.framework.c_api_util as c_api_util

def RegisterWatcherOnlyOnce(watcher):
    return c_api_util.RegisterWatcherOnlyOnce(watcher)

def IsOpTypeCaseCpuSupportOnly(op_type_case):
    return c_api_util.IsOpTypeCaseCpuSupportOnly(op_type_case)

def IsEnvInited():
    return c_api_util.IsEnvInited()

def InitEnv(env_proto):
    return c_api_util.InitEnv(env_proto)

def DestroyEnv():
    return c_api_util.DestroyEnv()

def IsSessionInited():
    return c_api_util.IsSessionInited()

def InitGlobalSession(config_proto):
    return c_api_util.InitGlobalSession(config_proto)

def DestroyGlobalSession():
    return c_api_util.DestroyGlobalSession()

def StartGlobalSession():
    return c_api_util.StartGlobalSession()

def StopGlobalSession():
    return c_api_util.StopGlobalSession()

def GetInterUserJobInfo():
    return c_api_util.GetInterUserJobInfo()

def LaunchJob(job_instance):
    return c_api_util.LaunchJob(job_instance)

def JobBuildAndInferCtx_Open(job_name):
    return c_api_util.JobBuildAndInferCtx_Open(job_name)

def JobBuildAndInferCtx_GetCurrentJobName():
    return c_api_util.JobBuildAndInferCtx_GetCurrentJobName()

def JobBuildAndInferCtx_Close():
    return c_api_util.JobBuildAndInferCtx_Close()

def CurJobBuildAndInferCtx_SetJobConf(job_config_proto):
    return c_api_util.CurJobBuildAndInferCtx_SetJobConf(job_config_proto)

def CurJobBuildAndInferCtx_Complete():
    return c_api_util.CurJobBuildAndInferCtx_Complete()

def CurJobBuildAndInferCtx_CheckAndCompleteUserOpConf(op_conf_proto):
    return c_api_util.CurJobBuildAndInferCtx_CheckAndCompleteUserOpConf(op_conf_proto)

def CurJobBuildAndInferCtx_AddAndInferOp(op_conf_proto, parallel_conf_proto):
    return c_api_util.CurJobBuildAndInferCtx_AddAndInferOp(op_conf_proto, parallel_conf_proto)

def CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf_proto, parallel_conf_proto):
    return c_api_util.CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf_proto, parallel_conf_proto)

def CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf_proto, parallel_conf_proto):
    return c_api_util.CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf_proto, parallel_conf_proto)

def CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn):
    return c_api_util.CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn)

def CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid):
    return c_api_util.CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid)

def CurJobBuildAndInferCtx_CheckJob():
    return c_api_util.CurJobBuildAndInferCtx_CheckJob()

def CurJobBuildAndInferCtx_HasJobConf():
    return c_api_util.CurJobBuildAndInferCtx_HasJobConf()

def JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn)

def JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn)

def JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, lbn, index):
    return c_api_util.JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, lbn, index)

def JobBuildAndInferCtx_MirroredBlobGetStaticShape(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_MirroredBlobGetStaticShape(job_name, lbn)

def JobBuildAndInferCtx_MirroredBlobGetDataType(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_MirroredBlobGetDataType(job_name, lbn)

def JobBuildAndInferCtx_MirroredBlobIsDynamic(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_MirroredBlobIsDynamic(job_name, lbn)

def JobBuildAndInferCtx_MirroredBlobDisableBoxing(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_MirroredBlobDisableBoxing(job_name, lbn)

def JobBuildAndInferCtx_MirroredBlobIsTensorList(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_MirroredBlobIsTensorList(job_name, lbn)

def JobBuildAndInferCtx_MirroredBlobGetBatchAxis(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_MirroredBlobGetBatchAxis(job_name, lbn)

def JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(job_name, lbn)

def JobBuildAndInferCtx_MirroredBlobGetParallelConfFromProducerView(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_MirroredBlobGetParallelConfFromProducerView(job_name, lbn)

def JobBuildAndInferCtx_GetStaticShape(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_GetStaticShape(job_name, lbn)

def JobBuildAndInferCtx_GetDataType(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_GetDataType(job_name, lbn)

def JobBuildAndInferCtx_IsDynamic(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_IsDynamic(job_name, lbn)

def JobBuildAndInferCtx_DisableBoxing(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_DisableBoxing(job_name, lbn)

def JobBuildAndInferCtx_IsTensorList(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_IsTensorList(job_name, lbn)

def JobBuildAndInferCtx_GetBatchAxis(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_GetBatchAxis(job_name, lbn)

def JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn)

def JobBuildAndInferCtx_GetParallelConfFromProducerView(job_name, lbn):
    return c_api_util.JobBuildAndInferCtx_GetParallelConfFromProducerView(job_name, lbn)

def GetMachine2DeviceIdListOFRecordFromParallelConf(parallel_conf):
    return c_api_util.GetMachine2DeviceIdListOFRecordFromParallelConf(parallel_conf)

def DeviceType4DeviceTag(device_tag):
    return c_api_util.DeviceType4DeviceTag(device_tag)

def GetFunctionConfigDef():
    return c_api_util.GetFunctionConfigDef()

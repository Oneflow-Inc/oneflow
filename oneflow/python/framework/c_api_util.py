from __future__ import absolute_import

import oneflow.core.common.error_pb2 as error_util
import oneflow.core.common.data_type_pb2 as dtype_util
from oneflow.core.job.inter_user_job_info_pb2 import InterUserJobInfo
import oneflow.core.job.job_set_pb2 as job_set_pb
import oneflow.core.job.env_pb2 as env_pb2
import oneflow.core.job.placement_pb2 as placment_util
import oneflow.core.record.record_pb2 as record_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
from google.protobuf import text_format
import oneflow.oneflow_internal as oneflow_internal
from oneflow.python.framework.job_build_and_infer_error import JobBuildAndInferError

def RegisterWatcherOnlyOnce(watcher):
    error_str = oneflow_internal.RegisterWatcherOnlyOnce(watcher)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def IsOpTypeCaseCpuSupportOnly(op_type_case):
    ret, error_str = oneflow_internal.IsOpTypeCaseCpuSupportOnly(op_type_case)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return ret

def IsEnvInited():
    return oneflow_internal.IsEnvInited()

def InitEnv(env_proto):
    assert(type(env_proto) is env_pb2.EnvProto)
    env_proto_str = text_format.MessageToString(env_proto)
    error_str = oneflow_internal.InitEnv(env_proto_str)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def DestroyEnv():
    error_str = oneflow_internal.DestroyEnv()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def IsSessionInited():
    return oneflow_internal.IsSessionInited()

def InitGlobalSession(config_proto):
    assert(type(config_proto) is job_set_pb.ConfigProto)
    config_proto_str = text_format.MessageToString(config_proto)
    error_str = oneflow_internal.InitGlobalSession(config_proto_str)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def DestroyGlobalSession():
    error_str = oneflow_internal.DestroyGlobalSession()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def StartGlobalSession():
    error_str = oneflow_internal.StartGlobalSession()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def StopGlobalSession():
    error_str = oneflow_internal.StopGlobalSession()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def GetInterUserJobInfo():
    inter_user_job_info, error_str = oneflow_internal.GetSerializedInterUserJobInfo()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return text_format.Parse(inter_user_job_info, InterUserJobInfo())

def LaunchJob(job_instance):
    error_str = oneflow_internal.LaunchJob(job_instance)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def JobBuildAndInferCtx_Open(job_name):
    job_name = str(job_name)
    error_str = oneflow_internal.JobBuildAndInferCtx_Open(job_name)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def JobBuildAndInferCtx_GetCurrentJobName():
    job_name, error_str = oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return job_name

def JobBuildAndInferCtx_Close():
    oneflow_internal.JobBuildAndInferCtx_Close()

def CurJobBuildAndInferCtx_SetJobConf(job_config_proto):
    serialized_job_conf = str(text_format.MessageToString(job_config_proto))
    error_str = oneflow_internal.CurJobBuildAndInferCtx_SetJobConf(serialized_job_conf)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def CurJobBuildAndInferCtx_CheckAndCompleteUserOpConf(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    AddDefaultVal = oneflow_internal.CurJobBuildAndInferCtx_CheckAndCompleteUserOpConf
    new_op_conf, error_str = AddDefaultVal(serialized_op_conf)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return text_format.Parse(new_op_conf, op_conf_util.OperatorConf())

def CurJobBuildAndInferCtx_AddAndInferOp(op_conf_proto, parallel_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    serialized_parallel_conf = str(text_format.MessageToString(parallel_conf_proto))
    add_and_infer = oneflow_internal.CurJobBuildAndInferCtx_AddAndInferOp
    error_str = add_and_infer(serialized_op_conf, serialized_parallel_conf)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn):
    lbn = str(lbn)
    error_str = oneflow_internal.CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    
def CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid):
    serialized = str(text_format.MessageToString(lbi_and_uuid))
    error_str = oneflow_internal.CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(serialized)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def CurJobBuildAndInferCtx_CheckJob():
    error_str = oneflow_internal.CurJobBuildAndInferCtx_CheckJob()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def CurJobBuildAndInferCtx_HasJobConf():
    has_job_conf, error_str = oneflow_internal.CurJobBuildAndInferCtx_HasJobConf()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return has_job_conf

def JobBuildAndInferCtx_GetStaticShape(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    axis_str, error_str = \
        oneflow_internal.JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    int_list = text_format.Parse(axis_str, record_util.Int64List())
    return tuple(map(int, int_list.value))

def JobBuildAndInferCtx_GetDataType(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    dtype, erro_str = oneflow_internal.JobBuildAndInferCtx_GetDataType(job_name, lbn)
    error = text_format.Parse(erro_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return int(dtype)

def JobBuildAndInferCtx_DisableBoxing(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error_str = oneflow_internal.JobBuildAndInferCtx_DisableBoxing(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return ret

def JobBuildAndInferCtx_GetBatchAxis(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    batch_axis_str, error_str = oneflow_internal.JobBuildAndInferCtx_GetBatchAxis(job_name, lbn)
    batch_axis = text_format.Parse(batch_axis_str, dtype_util.OptInt64())
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    if batch_axis.HasField("value"): return batch_axis.value
    return None

def JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    split_axis_str, error_str = \
        oneflow_internal.JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn)
    split_axis = text_format.Parse(split_axis_str, dtype_util.OptInt64())
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    if split_axis.HasField("value"): return split_axis.value
    return None

def JobBuildAndInferCtx_GetParallelConfFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    GetParallelConf = oneflow_internal.JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView
    parallel_conf, error_str = GetParallelConf(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return text_format.Parse(parallel_conf, placment_util.ParallelConf())

def GetMachine2DeviceIdListOFRecordFromParallelConf(parallel_conf):
    serialized_parallel_conf = str(text_format.MessageToString(parallel_conf))
    ofrecord, error_str = \
        oneflow_internal.GetMachine2DeviceIdListOFRecordFromParallelConf(serialized_parallel_conf)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return text_format.Parse(ofrecord, record_util.OFRecord())

def DeviceType4DeviceTag(device_tag):
    device_tag = str(device_tag)
    device_type, error_str = oneflow_internal.DeviceType4DeviceTag(device_tag)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return device_type

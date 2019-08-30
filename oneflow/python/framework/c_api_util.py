from __future__ import absolute_import

import oneflow.core.common.error_pb2 import error_util
from oneflow.core.job.inter_user_job_info_pb2 import InterUserJobInfo
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.core.job.placement_pb2 as placment_util
import oneflow.core.record.record_pb2 as record_util
from google.protobuf import text_format
import oneflow.oneflow_internal as oneflow_internal
import oneflow.python.framework.runtime_context as runtime_ctx

def IsOpTypeCaseCpuSupportOnly(op_type_case):
    return oneflow_internal.IsOpTypeCaseCpuSupportOnly(op_type_case)

def Init(config_proto):
    assert(type(config_proto) is job_set_util.ConfigProto)
    config_proto_str = text_format.MessageToString(config_proto)
    oneflow_internal.InitBySerializedConfigProto(config_proto_str)

def InitGlobalOneflowByJobSet(job_set):
    assert(type(job_set) is job_set_util.JobSet)
    job_set_str = text_format.MessageToString(job_set)
    oneflow_internal.InitGlobalOneflowBySerializedJobSet(job_set_str)

def GetInterUserJobInfo():
    return text_format.Parse(oneflow_internal.GetSerializedInterUserJobInfo(), InterUserJobInfo())

def LaunchJob(job_instance):
    for pre_launch_callback in runtime_ctx.job_instance_pre_launch_callbacks:
        pre_launch_callback(job_instance)
    for post_finish_callback in runtime_ctx.job_instance_post_finish_callbacks:
        job_instance.AddPostFinishCallback(post_finish_callback)
    oneflow_internal.LaunchJob(job_instance)

def DestroyGlobalOneflow():
    oneflow_internal.DestroyGlobalOneflow()

def DestroyGlobalEnvironment():
    oneflow_internal.DestroyGlobalEnvironment()

def JobBuildAndInferCtx_NewAndEnter(job_name):
    serialized_error = oneflow_internal.JobBuildAndInferCtx_NewAndEnter(job_name)
    error = text_format.Parse(serialized_error, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def JobBuildAndInferCtx_GetCurrentJobName():
    pair = oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    error = text_format.Parse(pair.second, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return pair.first

def JobBuildAndInferCtx_Leave():
    oneflow_internal.JobBuildAndInferCtx_Leave()

def CurJobBuildAndInferCtx_SetJobConf(job_config_proto):
    serialized_job_conf = text_format.MessageToString(job_config_proto)
    serialized_error = oneflow_internal.CurJobBuildAndInferCtx_SetJobConf(serialized_job_conf)
    error = text_format.Parse(serialized_error, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def CurJobBuildAndInferCtx_AddAndInferInputOp(op_conf_proto):
    serialized_op_conf = text_format.MessageToString(op_conf_proto)
    serialized_error = oneflow_internal.CurJobBuildAndInferCtx_AddAndInferInputOp(serialized_op_conf)
    error = text_format.Parse(serialized_error, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def CurJobBuildAndInferCtx_AddAndInferNonInputOp(op_conf_proto):
    serialized_op_conf = text_format.MessageToString(op_conf_proto)
    error_str = oneflow_internal.CurJobBuildAndInferCtx_AddAndInferNonInputOp(serialized_op_conf)
    error = text_format.Parse(error_str, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn):
    error_str = oneflow_internal.CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn)
    error = text_format.Parse(error_str, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

def CurJobBuildAndInferCtx_HasJobConf():
    pair =  oneflow_internal.CurJobBuildAndInferCtx_HasJobConf()
    error = text_format.Parse(pair.second, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return pair.first

def JobBuildAndInferCtx_GetStaticShape(job_name, lbn):
    pair = oneflow_internal.JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(job_name, lbn)
    error = text_format.Parse(pair.second, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    id_list = text_format.Parse(pair.first, record_util.Int64List())
    return tuple(id_list.value)

def JobBuildAndInferCtx_GetDataType(job_name, lbn):
    pair = oneflow_internal.JobBuildAndInferCtx_GetDataType(job_name, lbn)
    error = text_format.Parse(pair.second, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return pair.first

def JobBuildAndInferCtx_GetHasSplitDimFromProducerView(job_name, lbn):
    pair = oneflow_internal.JobBuildAndInferCtx_GetHasSplitDimFromProducerView(job_name, lbn)
    error = text_format.Parse(pair.second, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return pair.first

def JobBuildAndInferCtx_GetSplitDimFromProducerView(job_name, lbn):
    pair = oneflow_internal.JobBuildAndInferCtx_GetSplitDimFromProducerView(job_name, lbn)
    error = text_format.Parse(pair.second, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return pair.first

def JobBuildAndInferCtx_GetParallelConfFromProducerView(job_name, lbn):
    pair = oneflow_internal.JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(job_name,
                                                                                          lbn)
    error = text_format.Parse(pair.second, error_util.Error())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return text_format.Parse(pair.first, placment_util.ParallelConf())

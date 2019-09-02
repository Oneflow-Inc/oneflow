from __future__ import absolute_import

import oneflow.core.common.error_pb2 as error_util
import oneflow.core.common.data_type_pb2 as dtype_util
from oneflow.core.job.inter_user_job_info_pb2 import InterUserJobInfo
import oneflow.core.job.job_set_pb2 as job_set_util
import oneflow.core.job.placement_pb2 as placment_util
import oneflow.core.record.record_pb2 as record_util
from google.protobuf import text_format
import oneflow.oneflow_internal as oneflow_internal
import oneflow.python.framework.runtime_context as runtime_ctx
from oneflow.python.framework.job_build_and_infer_error import JobBuildAndInferError

def IsOpTypeCaseCpuSupportOnly(op_type_case):
    return oneflow_internal.IsOpTypeCaseCpuSupportOnly(op_type_case)

def Init(config_proto):
    assert(type(config_proto) is job_set_util.ConfigProto)
    config_proto_str = text_format.MessageToString(config_proto)
    oneflow_internal.InitBySerializedConfigProto(config_proto_str)

def InitGlobalOneflow():
    serialized_error = oneflow_internal.InitGlobalOneflow()
    error = text_format.Parse(serialized_error, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

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

def JobBuildAndInferCtx_Open(job_name):
    job_name = str(job_name)
    serialized_error = oneflow_internal.JobBuildAndInferCtx_Open(job_name)
    error = text_format.Parse(serialized_error, error_util.ErrorProto())
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
    serialized_error = oneflow_internal.CurJobBuildAndInferCtx_SetJobConf(serialized_job_conf)
    error = text_format.Parse(serialized_error, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)

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

def CurJobBuildAndInferCtx_AddPlacementGroup(placement_group_proto):
    serialized_placement_grp = str(text_format.MessageToString(placement_group_proto))
    error_str = oneflow_internal.CurJobBuildAndInferCtx_AddPlacementGroup(serialized_placement_grp)
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

def JobBuildAndInferCtx_GetBatchAxis(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    batch_axis_str, error_str = oneflow_internal.JobBuildAndInferCtx_GetBatchAxis(job_name, lbn)
    batch_axis = text_format.Parse(batch_axis_str, dtype_util.OptInt64())
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    if batch_axis.HasField("value"): return batch_axis.value
    return None

def JobBuildAndInferCtx_GetHasSplitAxisFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    has_split_axis, error_str = \
        oneflow_internal.JobBuildAndInferCtx_GetHasSplitAxisFromProducerView(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return has_split_axis

def JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    split_axis, error_str = \
        oneflow_internal.JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return split_axis

def JobBuildAndInferCtx_GetParallelConfFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    parallel_conf, error_str = oneflow_internal.JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return text_format.Parse(parallel_conf, placment_util.ParallelConf())

def DeviceType4DeviceTag(device_tag):
    device_tag = str(device_tag)
    device_type, error_str = oneflow_internal.DeviceType4DeviceTag(device_tag)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"): raise JobBuildAndInferError(error)
    return device_type


"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

from google.protobuf import text_format

import oneflow.core.common.data_type_pb2 as dtype_util
import oneflow.core.common.error_pb2 as error_util
import oneflow.core.job.env_pb2 as env_pb2
import oneflow.core.job.job_set_pb2 as job_set_pb
import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.core.operator.op_attribute_pb2 as op_attribute_pb
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.record.record_pb2 as record_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.oneflow_internal as oneflow_internal
from oneflow.core.framework.config_def_pb2 import ConfigDef
from oneflow.core.job.inter_user_job_info_pb2 import InterUserJobInfo
from oneflow.python.framework.job_build_and_infer_error import JobBuildAndInferError


def RegisterWatcherOnlyOnce(watcher):
    error_str = oneflow_internal.RegisterWatcherOnlyOnce(watcher)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def RegisterForeignCallbackOnlyOnce(callback):
    error_str = oneflow_internal.RegisterForeignCallbackOnlyOnce(callback)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def IsOpTypeCaseCpuSupportOnly(op_type_case):
    ret, error_str = oneflow_internal.IsOpTypeCaseCpuSupportOnly(op_type_case)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return ret


def IsOpTypeNameCpuSupportOnly(op_type_name):
    ret, error_str = oneflow_internal.IsOpTypeNameCpuSupportOnly(op_type_name)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return ret


def CurrentResource():
    resource, error_str = oneflow_internal.CurrentResource()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(resource, resource_util.Resource())


def EnvResource():
    resource, error_str = oneflow_internal.EnvResource()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(resource, resource_util.Resource())


def EnableEagerEnvironment(enable_eager_execution):
    return oneflow_internal.EnableEagerEnvironment(enable_eager_execution)


def EagerExecutionEnabled():
    return oneflow_internal.EagerExecutionEnabled()


def IsEnvInited():
    return oneflow_internal.IsEnvInited()


def InitEnv(env_proto):
    assert type(env_proto) is env_pb2.EnvProto
    env_proto_str = text_format.MessageToString(env_proto)
    error_str = oneflow_internal.InitEnv(env_proto_str)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def DestroyEnv():
    error_str = oneflow_internal.DestroyEnv()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def IsSessionInited():
    return oneflow_internal.IsSessionInited()


def InitGlobalSession(config_proto):
    assert type(config_proto) is job_set_pb.ConfigProto
    config_proto_str = text_format.MessageToString(config_proto)
    error_str = oneflow_internal.InitGlobalSession(config_proto_str)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def DestroyGlobalSession():
    error_str = oneflow_internal.DestroyGlobalSession()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def StartGlobalSession():
    error_str = oneflow_internal.StartGlobalSession()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def StopGlobalSession():
    error_str = oneflow_internal.StopGlobalSession()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def GetInterUserJobInfo():
    inter_user_job_info, error_str = oneflow_internal.GetSerializedInterUserJobInfo()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(inter_user_job_info, InterUserJobInfo())


def LaunchJob(job_instance):
    error_str = oneflow_internal.LaunchJob(job_instance)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def JobBuildAndInferCtx_Open(job_name):
    job_name = str(job_name)
    error_str = oneflow_internal.JobBuildAndInferCtx_Open(job_name)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def JobBuildAndInferCtx_GetCurrentJobName():
    job_name, error_str = oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return job_name


def JobBuildAndInferCtx_Close():
    oneflow_internal.JobBuildAndInferCtx_Close()


def CurJobBuildAndInferCtx_SetJobConf(job_config_proto):
    serialized_job_conf = str(text_format.MessageToString(job_config_proto))
    error_str = oneflow_internal.CurJobBuildAndInferCtx_SetJobConf(serialized_job_conf)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def CurJobBuildAndInferCtx_SetTrainConf(train_config_proto):
    serialized_train_conf = str(text_format.MessageToString(train_config_proto))
    error_str = oneflow_internal.CurJobBuildAndInferCtx_SetTrainConf(
        serialized_train_conf
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def CurJobBuildAndInferCtx_Complete():
    error_str = oneflow_internal.CurJobBuildAndInferCtx_Complete()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def InferOpConf(op_conf_proto, upstream_signature):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    serialized_upstream_sig = str(text_format.MessageToString(upstream_signature))
    op_attribute_str, error_str = oneflow_internal.InferOpConf(
        serialized_op_conf, serialized_upstream_sig,
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def GetOpParallelSymbolId(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    symbol_id, error_str = oneflow_internal.GetOpParallelSymbolId(serialized_op_conf)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return symbol_id


def GetUserOpAttrType(op_type_name, attr_name):
    attr_type, error_str = oneflow_internal.GetUserOpAttrType(op_type_name, attr_name)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return attr_type


def CheckAndCompleteUserOpConf(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    new_op_conf, error_str = oneflow_internal.CheckAndCompleteUserOpConf(
        serialized_op_conf
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(new_op_conf, op_conf_util.OperatorConf())


def CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    add_and_infer = oneflow_internal.CurJobBuildAndInferCtx_AddAndInferConsistentOp
    op_attribute_str, error_str = add_and_infer(serialized_op_conf)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    add_and_infer = oneflow_internal.CurJobBuildAndInferCtx_AddAndInferMirroredOp
    op_attribute_str, error_str = add_and_infer(serialized_op_conf)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn):
    lbn = str(lbn)
    error_str = oneflow_internal.CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid):
    serialized = str(text_format.MessageToString(lbi_and_uuid))
    error_str = oneflow_internal.CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(
        serialized
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def CurJobBuildAndInferCtx_CheckJob():
    error_str = oneflow_internal.CurJobBuildAndInferCtx_CheckJob()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def CurJobBuildAndInferCtx_HasJobConf():
    has_job_conf, error_str = oneflow_internal.CurJobBuildAndInferCtx_HasJobConf()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return has_job_conf


def JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error_str = oneflow_internal.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return ret


def JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error_str = oneflow_internal.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(
        job_name, lbn
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return ret


def JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, lbn, index):
    job_name = str(job_name)
    lbn = str(lbn)
    (
        ret,
        error_str,
    ) = oneflow_internal.JobBuildAndInferCtx_MirroredBlobGetSerializedSubLbi(
        job_name, lbn, index
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(ret, logical_blob_id_util.LogicalBlobId())


def JobBuildAndInferCtx_MirroredBlobGetStaticShape(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    get_shape = (
        oneflow_internal.JobBuildAndInferCtx_MirroredBlobGetSerializedIdListAsStaticShape
    )
    axis_str, error_str = get_shape(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    int_list = text_format.Parse(axis_str, record_util.Int64List())
    return tuple(map(int, int_list.value))


def JobBuildAndInferCtx_MirroredBlobGetDataType(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    dtype, erro_str = oneflow_internal.JobBuildAndInferCtx_MirroredBlobGetDataType(
        job_name, lbn
    )
    error = text_format.Parse(erro_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return int(dtype)


def JobBuildAndInferCtx_MirroredBlobIsDynamic(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error_str = oneflow_internal.JobBuildAndInferCtx_MirroredBlobIsDynamic(
        job_name, lbn
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return ret


def JobBuildAndInferCtx_MirroredBlobDisableBoxing(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error_str = oneflow_internal.JobBuildAndInferCtx_MirroredBlobDisableBoxing(
        job_name, lbn
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return ret


def JobBuildAndInferCtx_MirroredBlobIsTensorList(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error_str = oneflow_internal.JobBuildAndInferCtx_MirroredBlobIsTensorList(
        job_name, lbn
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return ret


def JobBuildAndInferCtx_MirroredBlobGetBatchAxis(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    (
        batch_axis_str,
        error_str,
    ) = oneflow_internal.JobBuildAndInferCtx_MirroredBlobGetBatchAxis(job_name, lbn)
    batch_axis = text_format.Parse(batch_axis_str, dtype_util.OptInt64())
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    if batch_axis.HasField("value"):
        return batch_axis.value
    return None


def JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    (
        split_axis_str,
        error_str,
    ) = oneflow_internal.JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(
        job_name, lbn
    )
    split_axis = text_format.Parse(split_axis_str, dtype_util.OptInt64())
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    if split_axis.HasField("value"):
        return split_axis.value
    return None


def JobBuildAndInferCtx_MirroredBlobGetParallelConfFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    GetParallelConf = (
        oneflow_internal.JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView
    )
    parallel_conf_str, error_str = GetParallelConf(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(parallel_conf_str, placement_pb.ParallelConf())


def JobBuildAndInferCtx_GetStaticShape(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    (
        axis_str,
        error_str,
    ) = oneflow_internal.JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(
        job_name, lbn
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    int_list = text_format.Parse(axis_str, record_util.Int64List())
    return tuple(map(int, int_list.value))


def JobBuildAndInferCtx_GetDataType(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    dtype, erro_str = oneflow_internal.JobBuildAndInferCtx_GetDataType(job_name, lbn)
    error = text_format.Parse(erro_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return int(dtype)


def JobBuildAndInferCtx_IsDynamic(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error_str = oneflow_internal.JobBuildAndInferCtx_IsDynamic(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return ret


def JobBuildAndInferCtx_DisableBoxing(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error_str = oneflow_internal.JobBuildAndInferCtx_DisableBoxing(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return ret


def JobBuildAndInferCtx_IsTensorList(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error_str = oneflow_internal.JobBuildAndInferCtx_IsTensorList(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return ret


def JobBuildAndInferCtx_GetBatchAxis(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    batch_axis_str, error_str = oneflow_internal.JobBuildAndInferCtx_GetBatchAxis(
        job_name, lbn
    )
    batch_axis = text_format.Parse(batch_axis_str, dtype_util.OptInt64())
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    if batch_axis.HasField("value"):
        return batch_axis.value
    return None


def JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    (
        split_axis_str,
        error_str,
    ) = oneflow_internal.JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn)
    split_axis = text_format.Parse(split_axis_str, dtype_util.OptInt64())
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    if split_axis.HasField("value"):
        return split_axis.value
    return None


def JobBuildAndInferCtx_GetParallelConfFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    GetParallelConf = (
        oneflow_internal.JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView
    )
    parallel_conf, error_str = GetParallelConf(job_name, lbn)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(parallel_conf, placement_pb.ParallelConf())


def GetMachine2DeviceIdListOFRecordFromParallelConf(parallel_conf):
    serialized_parallel_conf = str(text_format.MessageToString(parallel_conf))
    (
        ofrecord,
        error_str,
    ) = oneflow_internal.GetMachine2DeviceIdListOFRecordFromParallelConf(
        serialized_parallel_conf
    )
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(ofrecord, record_util.OFRecord())


def GetFunctionConfigDef():
    func_config_def, error_str = oneflow_internal.GetFunctionConfigDef()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(func_config_def, ConfigDef())


def RunLogicalInstruction(vm_instruction_list, eager_symbol_list):
    instructions = str(text_format.MessageToString(vm_instruction_list))
    symbols = str(text_format.MessageToString(eager_symbol_list))
    error_str = oneflow_internal.RunLogicalInstruction(instructions, symbols)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def RunPhysicalInstruction(vm_instruction_list, eager_symbol_list):
    instructions = str(text_format.MessageToString(vm_instruction_list))
    symbols = str(text_format.MessageToString(eager_symbol_list))
    error_str = oneflow_internal.RunPhysicalInstruction(instructions, symbols)
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)


def CurrentMachineId():
    machine_id, error_str = oneflow_internal.CurrentMachineId()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return machine_id


def NewLogicalObjectId():
    object_id, error_str = oneflow_internal.NewLogicalObjectId()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return object_id


def NewLogicalSymbolId():
    object_id, error_str = oneflow_internal.NewLogicalSymbolId()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return object_id


def NewPhysicalObjectId():
    object_id, error_str = oneflow_internal.NewPhysicalObjectId()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return object_id


def NewPhysicalSymbolId():
    object_id, error_str = oneflow_internal.NewPhysicalSymbolId()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return object_id


def GetJobSet():
    job_set, error_str = oneflow_internal.GetSerializedJobSet()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return text_format.Parse(job_set, job_set_pb.JobSet())


def GetStructureGraph():
    structure_graph, error_str = oneflow_internal.GetSerializedStructureGraph()
    error = text_format.Parse(error_str, error_util.ErrorProto())
    if error.HasField("error_type"):
        raise JobBuildAndInferError(error)
    return structure_graph

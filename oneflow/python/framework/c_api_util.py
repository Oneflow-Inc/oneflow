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
from oneflow.core.framework.config_def_pb2 import ConfigDef
from oneflow.core.job.inter_user_job_info_pb2 import InterUserJobInfo
from oneflow.python.framework.job_build_and_infer_cfg_error import (
    JobBuildAndInferCfgError,
)
import oneflow
import oneflow_api.oneflow.core.common.error as error_cfg
import oneflow_api.oneflow.core.job.placement as placement_cfg

oneflow_api = oneflow.oneflow_api


def RegisterForeignCallbackOnlyOnce(callback):
    error = oneflow_api.RegisterForeignCallbackOnlyOnce(callback)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def RegisterWatcherOnlyOnce(watcher):
    error = oneflow_api.RegisterWatcherOnlyOnce(watcher)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def IsOpTypeCaseCpuSupportOnly(op_type_case):
    ret, error = oneflow_api.IsOpTypeCaseCpuSupportOnly(op_type_case)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return ret


def IsOpTypeNameCpuSupportOnly(op_type_name):
    ret, error = oneflow_api.IsOpTypeNameCpuSupportOnly(op_type_name)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return ret


def CurrentResource():
    resource, error = oneflow_api.CurrentResource()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(resource, resource_util.Resource())


def EnvResource():
    resource, error = oneflow_api.EnvResource()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(resource, resource_util.Resource())


def EnableEagerEnvironment(enable_eager_execution):
    error = oneflow_api.EnableEagerEnvironment(enable_eager_execution)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def EagerExecutionEnabled():
    return oneflow_api.EagerExecutionEnabled()


def IsEnvInited():
    is_env_inited, error = oneflow_api.IsEnvInited()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return is_env_inited


def InitEnv(env_proto):
    assert type(env_proto) is env_pb2.EnvProto
    env_proto_str = text_format.MessageToString(env_proto)
    error = oneflow_api.InitEnv(env_proto_str)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def DestroyEnv():
    error = oneflow_api.DestroyEnv()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def IsSessionInited():
    is_sess_inited, error = oneflow_api.IsSessionInited()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return is_sess_inited


def InitLazyGlobalSession(config_proto):
    assert type(config_proto) is job_set_pb.ConfigProto
    config_proto_str = text_format.MessageToString(config_proto)
    error = oneflow_api.InitLazyGlobalSession(config_proto_str)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def DestroyLazyGlobalSession():
    error = oneflow_api.DestroyLazyGlobalSession()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def StartLazyGlobalSession():
    error = oneflow_api.StartLazyGlobalSession()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def StopLazyGlobalSession():
    error = oneflow_api.StopLazyGlobalSession()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def GetInterUserJobInfo():
    inter_user_job_info, error = oneflow_api.GetSerializedInterUserJobInfo()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(inter_user_job_info, InterUserJobInfo())


def LaunchJob(job_instance):
    error = oneflow_api.LaunchJob(job_instance)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def JobBuildAndInferCtx_Open(job_name):
    job_name = str(job_name)
    error = oneflow_api.JobBuildAndInferCtx_Open(job_name)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def JobBuildAndInferCtx_GetCurrentJobName():
    job_name, error = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return job_name


def JobBuildAndInferCtx_Close():
    error = oneflow_api.JobBuildAndInferCtx_Close()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def CurJobBuildAndInferCtx_SetJobConf(job_config_proto):
    serialized_job_conf = str(job_config_proto)
    error = oneflow_api.CurJobBuildAndInferCtx_SetJobConf(serialized_job_conf)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def CurJobBuildAndInferCtx_SetTrainConf(train_config_proto):
    serialized_train_conf = str(text_format.MessageToString(train_config_proto))
    error = oneflow_api.CurJobBuildAndInferCtx_SetTrainConf(serialized_train_conf)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def CurJobBuildAndInferCtx_Complete():
    error = oneflow_api.CurJobBuildAndInferCtx_Complete()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def InferOpConf(op_conf_proto, upstream_signature):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    serialized_upstream_sig = str(text_format.MessageToString(upstream_signature))
    op_attribute_str, error = oneflow_api.InferOpConf(
        serialized_op_conf, serialized_upstream_sig,
    )
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def IsInterfaceOpConf(op_conf):
    op_type_field = op_conf.WhichOneof("op_type")
    field_number = op_conf_util.OperatorConf.DESCRIPTOR.fields_by_name[
        op_type_field
    ].number
    res, error = oneflow_api.IsInterfaceOpTypeCase(field_number)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return res


def GetOpParallelSymbolId(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    symbol_id, error = oneflow_api.GetOpParallelSymbolId(serialized_op_conf)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return symbol_id


def GetUserOpAttrType(op_type_name, attr_name):
    attr_type, error = oneflow_api.GetUserOpAttrType(op_type_name, attr_name)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return attr_type


def CheckAndCompleteUserOpConf(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    new_op_conf, error = oneflow_api.CheckAndCompleteUserOpConf(serialized_op_conf)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(new_op_conf, op_conf_util.OperatorConf())


def CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    add_and_infer = oneflow_api.CurJobBuildAndInferCtx_AddAndInferConsistentOp
    op_attribute_str, error = add_and_infer(serialized_op_conf)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    add_and_infer = oneflow_api.CurJobBuildAndInferCtx_AddAndInferMirroredOp
    op_attribute_str, error = add_and_infer(serialized_op_conf)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn):
    lbn = str(lbn)
    error = oneflow_api.CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid):
    serialized = str(text_format.MessageToString(lbi_and_uuid))
    error = oneflow_api.CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(serialized)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def CurJobBuildAndInferCtx_CheckJob():
    error = oneflow_api.CurJobBuildAndInferCtx_CheckJob()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def CurJobBuildAndInferCtx_HasJobConf():
    has_job_conf, error = oneflow_api.CurJobBuildAndInferCtx_HasJobConf()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return has_job_conf


def JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error = oneflow_api.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return ret


def JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error = oneflow_api.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return ret


def JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, lbn, index):
    job_name = str(job_name)
    lbn = str(lbn)
    (ret, error,) = oneflow_api.JobBuildAndInferCtx_MirroredBlobGetSerializedSubLbi(
        job_name, lbn, index
    )
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(ret, logical_blob_id_util.LogicalBlobId())


def JobBuildAndInferCtx_MirroredBlobGetStaticShape(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    get_shape = (
        oneflow_api.JobBuildAndInferCtx_MirroredBlobGetSerializedIdListAsStaticShape
    )
    axis_str, error = get_shape(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    int_list = text_format.Parse(axis_str, record_util.Int64List())
    return tuple(map(int, int_list.value))


def JobBuildAndInferCtx_MirroredBlobGetDataType(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    dtype, error = oneflow_api.JobBuildAndInferCtx_MirroredBlobGetDataType(
        job_name, lbn
    )
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return int(dtype)


def JobBuildAndInferCtx_MirroredBlobIsDynamic(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error = oneflow_api.JobBuildAndInferCtx_MirroredBlobIsDynamic(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return ret


def JobBuildAndInferCtx_MirroredBlobIsTensorList(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error = oneflow_api.JobBuildAndInferCtx_MirroredBlobIsTensorList(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return ret


def JobBuildAndInferCtx_MirroredBlobGetBatchAxis(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    (batch_axis_str, error,) = oneflow_api.JobBuildAndInferCtx_MirroredBlobGetBatchAxis(
        job_name, lbn
    )
    batch_axis = text_format.Parse(batch_axis_str, dtype_util.OptInt64())
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    if batch_axis.HasField("value"):
        return batch_axis.value
    return None


def JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    (
        split_axis_str,
        error,
    ) = oneflow_api.JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(
        job_name, lbn
    )
    split_axis = text_format.Parse(split_axis_str, dtype_util.OptInt64())
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    if split_axis.HasField("value"):
        return split_axis.value
    return None


def JobBuildAndInferCtx_MirroredBlobGetParallelConfFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    GetParallelConf = (
        oneflow_api.JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView
    )
    parallel_conf_str, error = GetParallelConf(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    parallel_conf = text_format.Parse(parallel_conf_str, placement_pb.ParallelConf())
    # TODO(oyy) change temporary transformation after python code migrated into cpp code
    parallel_conf_cfg = placement_cfg.ParallelConf()
    parallel_conf_cfg.set_device_tag(parallel_conf.device_tag)
    for device_name in parallel_conf.device_name:
        parallel_conf_cfg.add_device_name(device_name)

    return parallel_conf_cfg


def JobBuildAndInferCtx_GetStaticShape(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    (
        axis_str,
        error,
    ) = oneflow_api.JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    int_list = text_format.Parse(axis_str, record_util.Int64List())
    return tuple(map(int, int_list.value))


def JobBuildAndInferCtx_GetDataType(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    dtype, error = oneflow_api.JobBuildAndInferCtx_GetDataType(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return int(dtype)


def JobBuildAndInferCtx_IsDynamic(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error = oneflow_api.JobBuildAndInferCtx_IsDynamic(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return ret


def JobBuildAndInferCtx_DisableBoxing(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error = oneflow_api.JobBuildAndInferCtx_DisableBoxing(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return ret


def JobBuildAndInferCtx_IsTensorList(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret, error = oneflow_api.JobBuildAndInferCtx_IsTensorList(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return ret


def JobBuildAndInferCtx_GetBatchAxis(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    batch_axis_str, error = oneflow_api.JobBuildAndInferCtx_GetBatchAxis(job_name, lbn)
    batch_axis = text_format.Parse(batch_axis_str, dtype_util.OptInt64())
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    if batch_axis.HasField("value"):
        return batch_axis.value
    return None


def JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    (
        split_axis_str,
        error,
    ) = oneflow_api.JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn)
    split_axis = text_format.Parse(split_axis_str, dtype_util.OptInt64())
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    if split_axis.HasField("value"):
        return split_axis.value
    return None


def JobBuildAndInferCtx_GetParallelConfFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    GetParallelConf = (
        oneflow_api.JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView
    )
    parallel_conf, error = GetParallelConf(job_name, lbn)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    parallel_conf = text_format.Parse(parallel_conf, placement_pb.ParallelConf())
    # TODO(oyy) change temporary transformation after python code migrated into cpp code
    parallel_conf_cfg = placement_cfg.ParallelConf()
    parallel_conf_cfg.set_device_tag(parallel_conf.device_tag)
    for device_name in parallel_conf.device_name:
        parallel_conf_cfg.add_device_name(device_name)

    return parallel_conf_cfg


def GetMachine2DeviceIdListOFRecordFromParallelConf(parallel_conf):
    serialized_parallel_conf = str(parallel_conf)
    (ofrecord, error) = oneflow_api.GetMachine2DeviceIdListOFRecordFromParallelConf(
        serialized_parallel_conf
    )
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(ofrecord, record_util.OFRecord())


def GetFunctionConfigDef():
    func_config_def, error = oneflow_api.GetFunctionConfigDef()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(func_config_def, ConfigDef())


def GetScopeConfigDef():
    scope_config_def, error = oneflow_api.GetScopeConfigDef()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(scope_config_def, ConfigDef())


def RunLogicalInstruction(vm_instruction_list, eager_symbol_list):
    symbols = str(text_format.MessageToString(eager_symbol_list))
    error = oneflow_api.vm.RunLogicalInstruction(vm_instruction_list, symbols)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def RunPhysicalInstruction(vm_instruction_list, eager_symbol_list):
    symbols = str(text_format.MessageToString(eager_symbol_list))
    error = oneflow_api.vm.RunPhysicalInstruction(vm_instruction_list, symbols)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)


def CurrentMachineId():
    machine_id, error = oneflow_api.CurrentMachineId()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return machine_id


def NewLogicalObjectId():
    object_id, error = oneflow_api.NewLogicalObjectId()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return object_id


def NewLogicalSymbolId():
    object_id, error = oneflow_api.NewLogicalSymbolId()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return object_id


def NewPhysicalObjectId():
    object_id, error = oneflow_api.NewPhysicalObjectId()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return object_id


def NewPhysicalSymbolId():
    object_id, error = oneflow_api.NewPhysicalSymbolId()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return object_id


def GetOpAttributes():
    op_attributes, error = oneflow_api.GetSerializedOpAttributes()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(op_attributes, op_attribute_pb.OpAttributeList())


def GetJobSet():
    job_set, error = oneflow_api.GetSerializedJobSet()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return text_format.Parse(job_set, job_set_pb.JobSet())


def GetStructureGraph():
    structure_graph, error = oneflow_api.GetSerializedStructureGraph()
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)
    return structure_graph


def LoadLibraryNow(lib_path):
    error = oneflow_api.LoadLibraryNow(lib_path)
    if error.has_error_type():
        raise JobBuildAndInferCfgError(error)

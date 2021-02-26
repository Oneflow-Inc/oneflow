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
import oneflow.core.job.job_pb2 as job_pb
import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.core.operator.op_attribute_pb2 as op_attribute_pb
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.record.record_pb2 as record_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
from oneflow.core.framework.config_def_pb2 import ConfigDef
from oneflow.core.job.inter_user_job_info_pb2 import InterUserJobInfo
import oneflow
import oneflow_api.oneflow.core.common.error as error_cfg
import oneflow_api.oneflow.core.job.placement as placement_cfg

oneflow_api = oneflow.oneflow_api


def CurrentResource():
    resource = oneflow_api.CurrentResource()
    return text_format.Parse(resource, resource_util.Resource())


def EnvResource():
    resource = oneflow_api.EnvResource()
    return text_format.Parse(resource, resource_util.Resource())


def InitEnv(env_proto):
    assert type(env_proto) is env_pb2.EnvProto
    env_proto_str = text_format.MessageToString(env_proto)
    oneflow_api.InitEnv(env_proto_str)


def InitLazyGlobalSession(config_proto):
    assert type(config_proto) is job_set_pb.ConfigProto
    config_proto_str = text_format.MessageToString(config_proto)
    oneflow_api.InitLazyGlobalSession(config_proto_str)


def GetInterUserJobInfo():
    inter_user_job_info = oneflow_api.GetSerializedInterUserJobInfo()
    return text_format.Parse(inter_user_job_info, InterUserJobInfo())


def JobBuildAndInferCtx_Open(job_name):
    job_name = str(job_name)
    oneflow_api.JobBuildAndInferCtx_Open(job_name)


def CurJobBuildAndInferCtx_SetJobConf(job_config_proto):
    serialized_job_conf = str(job_config_proto)
    oneflow_api.CurJobBuildAndInferCtx_SetJobConf(serialized_job_conf)


def CurJobBuildAndInferCtx_SetTrainConf(train_config_proto):
    serialized_train_conf = str(text_format.MessageToString(train_config_proto))
    oneflow_api.CurJobBuildAndInferCtx_SetTrainConf(serialized_train_conf)


def InferOpConf(op_conf_proto, upstream_signature):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    serialized_upstream_sig = str(text_format.MessageToString(upstream_signature))
    op_attribute_str = oneflow_api.InferOpConf(
        serialized_op_conf, serialized_upstream_sig,
    )
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def IsInterfaceOpConf(op_conf):
    op_type_field = op_conf.WhichOneof("op_type")
    field_number = op_conf_util.OperatorConf.DESCRIPTOR.fields_by_name[
        op_type_field
    ].number
    return oneflow_api.IsInterfaceOpTypeCase(field_number)


def GetOpParallelSymbolId(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    return oneflow_api.GetOpParallelSymbolId(serialized_op_conf)


def CheckAndCompleteUserOpConf(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    new_op_conf = oneflow_api.CheckAndCompleteUserOpConf(serialized_op_conf)
    return text_format.Parse(new_op_conf, op_conf_util.OperatorConf())


def CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    add_and_infer = oneflow_api.CurJobBuildAndInferCtx_AddAndInferConsistentOp
    op_attribute_str = add_and_infer(serialized_op_conf)
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    add_and_infer = oneflow_api.CurJobBuildAndInferCtx_AddAndInferMirroredOp
    op_attribute_str = add_and_infer(serialized_op_conf)
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn):
    lbn = str(lbn)
    oneflow_api.CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn)


def CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid):
    serialized = str(text_format.MessageToString(lbi_and_uuid))
    oneflow_api.CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(serialized)


def JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    return oneflow_api.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn)


def JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    return oneflow_api.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn)


def JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, lbn, index):
    job_name = str(job_name)
    lbn = str(lbn)
    ret = oneflow_api.JobBuildAndInferCtx_MirroredBlobGetSerializedSubLbi(
        job_name, lbn, index
    )
    return text_format.Parse(ret, logical_blob_id_util.LogicalBlobId())


def JobBuildAndInferCtx_GetStaticShape(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    axis_str = oneflow_api.JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(
        job_name, lbn
    )
    int_list = text_format.Parse(axis_str, record_util.Int64List())
    return tuple(map(int, int_list.value))


def JobBuildAndInferCtx_GetDataType(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    dtype = oneflow_api.JobBuildAndInferCtx_GetDataType(job_name, lbn)
    return int(dtype)


def JobBuildAndInferCtx_IsDynamic(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret = oneflow_api.JobBuildAndInferCtx_IsDynamic(job_name, lbn)
    return ret


def JobBuildAndInferCtx_DisableBoxing(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret = oneflow_api.JobBuildAndInferCtx_DisableBoxing(job_name, lbn)
    return ret


def JobBuildAndInferCtx_IsTensorList(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret = oneflow_api.JobBuildAndInferCtx_IsTensorList(job_name, lbn)
    return ret


def JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    split_axis_str = oneflow_api.JobBuildAndInferCtx_GetSplitAxisFromProducerView(
        job_name, lbn
    )
    split_axis = text_format.Parse(split_axis_str, dtype_util.OptInt64())
    if split_axis.HasField("value"):
        return split_axis.value
    return None


def JobBuildAndInferCtx_GetParallelConfFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    GetParallelConf = (
        oneflow_api.JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView
    )
    parallel_conf = GetParallelConf(job_name, lbn)
    parallel_conf = text_format.Parse(parallel_conf, placement_pb.ParallelConf())
    # TODO(oyy) change temporary transformation after python code migrated into cpp code
    parallel_conf_cfg = placement_cfg.ParallelConf()
    parallel_conf_cfg.set_device_tag(parallel_conf.device_tag)
    for device_name in parallel_conf.device_name:
        parallel_conf_cfg.add_device_name(device_name)

    return parallel_conf_cfg


def GetMachine2DeviceIdListOFRecordFromParallelConf(parallel_conf):
    serialized_parallel_conf = str(parallel_conf)
    ofrecord = oneflow_api.GetMachine2DeviceIdListOFRecordFromParallelConf(
        serialized_parallel_conf
    )
    return text_format.Parse(ofrecord, record_util.OFRecord())


def GetFunctionConfigDef():
    func_config_def = oneflow_api.GetFunctionConfigDef()
    return text_format.Parse(func_config_def, ConfigDef())


def GetScopeConfigDef():
    scope_config_def = oneflow_api.GetScopeConfigDef()
    return text_format.Parse(scope_config_def, ConfigDef())


def GetOpAttributes():
    op_attributes = oneflow_api.GetSerializedOpAttributes()
    return text_format.Parse(op_attributes, op_attribute_pb.OpAttributeList())


def GetJobSet():
    job_set = oneflow_api.GetSerializedJobSet()
    return text_format.Parse(job_set, job_set_pb.JobSet())


def GetCurrentJob():
    serialized_job = oneflow_api.GetSerializedCurrentJob()
    return text_format.Parse(serialized_job, job_pb.Job())

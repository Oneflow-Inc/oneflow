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
from google.protobuf import text_format

import oneflow
import oneflow.core.common.data_type_pb2 as dtype_util
import oneflow.core.common.error_pb2 as error_util
import oneflow.core.job.env_pb2 as env_pb2
import oneflow.core.job.job_pb2 as job_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb
import oneflow.core.job.job_set_pb2 as job_set_pb
import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.core.job.resource_pb2 as resource_util
import oneflow.core.operator.op_attribute_pb2 as op_attribute_pb
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.record.record_pb2 as record_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
from oneflow.core.framework.config_def_pb2 import ConfigDef
from oneflow.core.job.inter_user_job_info_pb2 import InterUserJobInfo
from oneflow.core.serving.saved_model_pb2 import SavedModel


def CurrentResource():
    resource = oneflow._oneflow_internal.CurrentResource()
    return text_format.Parse(resource, resource_util.Resource())


def EnvResource():
    resource = oneflow._oneflow_internal.EnvResource()
    return text_format.Parse(resource, resource_util.Resource())


def GetEnvContext(env_proto):
    assert type(env_proto) is env_pb2.EnvProto
    env_proto_str = text_format.MessageToString(env_proto)
    env_ctx = oneflow._oneflow_internal.EnvContext(env_proto_str)
    return env_ctx


def InitLazyGlobalSession(config_proto):
    assert type(config_proto) is job_set_pb.ConfigProto
    config_proto_str = text_format.MessageToString(config_proto)
    oneflow._oneflow_internal.InitLazyGlobalSession(config_proto_str)


def GetInterUserJobInfo():
    inter_user_job_info = oneflow._oneflow_internal.GetSerializedInterUserJobInfo()
    ret = InterUserJobInfo()
    ret.ParseFromString(inter_user_job_info)
    return ret


def JobBuildAndInferCtx_Open(job_name):
    job_name = str(job_name)
    oneflow._oneflow_internal.JobBuildAndInferCtx_Open(job_name)


def CurJobBuildAndInferCtx_SetJobConf(job_config_proto):
    assert type(job_config_proto) is job_conf_pb.JobConfigProto, type(job_config_proto)
    job_config_proto_str = text_format.MessageToString(job_config_proto)
    oneflow._oneflow_internal.CurJobBuildAndInferCtx_SetJobConf(job_config_proto_str)


def CurJobBuildAndInferCtx_SetTrainConf(train_config):
    assert type(train_config) is job_conf_pb.TrainConf
    train_config_str = text_format.MessageToString(train_config)
    oneflow._oneflow_internal.CurJobBuildAndInferCtx_SetTrainConf(train_config_str)


def InferOpConf(op_conf_proto, upstream_signature):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    serialized_upstream_sig = str(text_format.MessageToString(upstream_signature))
    op_attribute_str = oneflow._oneflow_internal.InferOpConf(
        serialized_op_conf, serialized_upstream_sig
    )
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def IsInterfaceOpConf(op_conf):
    op_type_field = op_conf.WhichOneof("op_type")
    field_number = op_conf_util.OperatorConf.DESCRIPTOR.fields_by_name[
        op_type_field
    ].number
    return oneflow._oneflow_internal.IsInterfaceOpTypeCase(field_number)


def GetOpParallelSymbolId(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    return oneflow._oneflow_internal.GetOpParallelSymbolId(serialized_op_conf)


def CheckAndCompleteUserOpConf(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    new_op_conf = oneflow._oneflow_internal.CheckAndCompleteUserOpConf(
        serialized_op_conf
    )
    return text_format.Parse(new_op_conf, op_conf_util.OperatorConf())


def CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    add_and_infer = (
        oneflow._oneflow_internal.CurJobBuildAndInferCtx_AddAndInferConsistentOp
    )
    op_attribute_str = add_and_infer(serialized_op_conf)
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf_proto):
    serialized_op_conf = str(text_format.MessageToString(op_conf_proto))
    add_and_infer = (
        oneflow._oneflow_internal.CurJobBuildAndInferCtx_AddAndInferMirroredOp
    )
    op_attribute_str = add_and_infer(serialized_op_conf)
    return text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())


def CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn):
    lbn = str(lbn)
    oneflow._oneflow_internal.CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn)


def CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_and_uuid):
    serialized = str(text_format.MessageToString(lbi_and_uuid))
    oneflow._oneflow_internal.CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(
        serialized
    )


def JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    return oneflow._oneflow_internal.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn)


def JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    return oneflow._oneflow_internal.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(
        job_name, lbn
    )


def JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, lbn, index):
    job_name = str(job_name)
    lbn = str(lbn)
    ret = oneflow._oneflow_internal.JobBuildAndInferCtx_MirroredBlobGetSerializedSubLbi(
        job_name, lbn, index
    )
    return text_format.Parse(ret, logical_blob_id_util.LogicalBlobId())


def JobBuildAndInferCtx_GetStaticShape(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    axis_str = oneflow._oneflow_internal.JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(
        job_name, lbn
    )
    int_list = text_format.Parse(axis_str, record_util.Int64List())
    return tuple(map(int, int_list.value))


def JobBuildAndInferCtx_GetDataType(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    dtype = oneflow._oneflow_internal.JobBuildAndInferCtx_GetDataType(job_name, lbn)
    return int(dtype)


def JobBuildAndInferCtx_IsDynamic(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret = oneflow._oneflow_internal.JobBuildAndInferCtx_IsDynamic(job_name, lbn)
    return ret


def JobBuildAndInferCtx_DisableBoxing(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    ret = oneflow._oneflow_internal.JobBuildAndInferCtx_DisableBoxing(job_name, lbn)
    return ret


def JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn):
    job_name = str(job_name)
    lbn = str(lbn)
    split_axis_str = oneflow._oneflow_internal.JobBuildAndInferCtx_GetSplitAxisFromProducerView(
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
        oneflow._oneflow_internal.JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView
    )
    serialized_parallel_conf = GetParallelConf(job_name, lbn)
    parallel_conf = text_format.Parse(
        serialized_parallel_conf, placement_pb.ParallelConf()
    )
    return parallel_conf


def GetMachine2DeviceIdListOFRecordFromParallelConf(parallel_conf):
    serialized_parallel_conf = str(parallel_conf)
    ofrecord = oneflow._oneflow_internal.GetMachine2DeviceIdListOFRecordFromParallelConf(
        serialized_parallel_conf
    )
    return text_format.Parse(ofrecord, record_util.OFRecord())


def GetFunctionConfigDef():
    func_config_def = oneflow._oneflow_internal.GetFunctionConfigDef()
    return text_format.Parse(func_config_def, ConfigDef())


def GetScopeConfigDef():
    scope_config_def = oneflow._oneflow_internal.GetScopeConfigDef()
    return text_format.Parse(scope_config_def, ConfigDef())


def GetInterfaceOpAttributes():
    op_attributes = oneflow._oneflow_internal.GetSerializedInterfaceOpAttributes()
    return text_format.Parse(op_attributes, op_attribute_pb.OpAttributeList())


def GetJobSet():
    job_set = oneflow._oneflow_internal.GetSerializedJobSet()
    ret = job_set_pb.JobSet()
    ret.ParseFromString(job_set)
    return ret


def GetCurrentJob():
    serialized_job = oneflow._oneflow_internal.GetSerializedCurrentJob()
    ret = job_pb.Job()
    ret.ParseFromString(serialized_job)
    return ret


def LoadSavedModel(saved_model_meta_file, is_prototxt_file):
    serialized_saved_model = oneflow._oneflow_internal.LoadSavedModel(
        saved_model_meta_file, is_prototxt_file
    )
    saved_model = text_format.Parse(serialized_saved_model, SavedModel())
    return saved_model

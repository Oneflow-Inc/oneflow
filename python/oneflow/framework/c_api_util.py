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


def JobBuildAndInferCtx_Open(job_name):
    job_name = str(job_name)
    oneflow._oneflow_internal.JobBuildAndInferCtx_Open(job_name)


def CurJobBuildAndInferCtx_SetJobConf(job_config_proto):
    assert type(job_config_proto) is job_conf_pb.JobConfigProto, type(job_config_proto)
    job_config_proto_str = text_format.MessageToString(job_config_proto)
    oneflow._oneflow_internal.CurJobBuildAndInferCtx_SetJobConf(job_config_proto_str)


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


def GetFunctionConfigDef():
    func_config_def = oneflow._oneflow_internal.GetFunctionConfigDef()
    return text_format.Parse(func_config_def, ConfigDef())


def GetScopeConfigDef():
    scope_config_def = oneflow._oneflow_internal.GetScopeConfigDef()
    return text_format.Parse(scope_config_def, ConfigDef())


def GetCurrentJob():
    serialized_job = oneflow._oneflow_internal.GetSerializedCurrentJob()
    ret = job_pb.Job()
    ret.ParseFromString(serialized_job)
    return ret

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

import oneflow.core.operator.op_attribute_pb2 as op_attribute_pb
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.boxing_util as boxing_util
import oneflow.python.framework.device_util as device_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.op_arg_util as op_arg_util
import oneflow.python.experimental.name_scope as name_scope
import oneflow.python.framework.session_context as session_ctx
import oneflow.core.job.placement_pb2 as placement_pb
from google.protobuf import text_format

import oneflow


def Interpret(op_attribute, parallel_conf, blob_register):
    if op_attribute.op_conf.HasField("cast_to_mirrored_conf"):
        return MirroredCast(op_attribute, blob_register)
    if op_attribute.op_conf.HasField("cast_from_mirrored_conf"):
        return MirroredCast(op_attribute, blob_register)
    if type(parallel_conf) is str:
        parallel_conf = text_format.Parse(parallel_conf, placement_pb.ParallelConf())
    else:
        assert isinstance(parallel_conf, placement_pb.ParallelConf)
    if op_attribute.op_conf.HasField("variable_conf"):
        return _FindOrCreateVarBlobObject(op_attribute, parallel_conf, blob_register)
    if op_attribute.op_conf.HasField("foreign_watch_conf"):
        return _Watch(op_attribute, parallel_conf, blob_register)
    return _NaiveInterpret(op_attribute, parallel_conf, blob_register)


def OpKernelCall(opkernel_object, op_attribute, blob_register):
    def BuildInstruction(builder):
        with blob_register.BnInOp2BlobObjectScope(op_attribute) as bn_in_op2blob_object:
            builder.StatefulCall(
                op_attribute,
                opkernel_object=opkernel_object,
                bn_in_op2blob_object=bn_in_op2blob_object,
            )

    vm_util.LogicalRun(BuildInstruction)


def MirroredCast(op_attribute, blob_register):
    def BuildInstruction(builder):
        with blob_register.BnInOp2BlobObjectScope(op_attribute) as bn_in_op2blob_object:
            in_blob_object = bn_in_op2blob_object["in"]
            parallel_desc_symbol = in_blob_object.parallel_desc_symbol
            op_arg_parallel_attr = op_arg_util.GetOpArgParallelAttribute(
                parallel_desc_symbol, op_attribute, "out"
            )
            out_blob_object = builder.MakeReferenceBlobObject(
                in_blob_object, op_arg_parallel_attr
            )
            bn_in_op2blob_object["out"] = out_blob_object

    vm_util.LogicalRun(BuildInstruction)


def _FindOrCreateVarBlobObject(op_attribute, parallel_conf, blob_register):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    name = name_scope.GetJobNameScopePrefix(job_name) + op_attribute.op_conf.name
    sess = session_ctx.GetDefaultSession()
    var_blob, _ = sess.TryGetVariableBlobOfJobFromStash(job_name, name)
    if var_blob is not None:
        blob_register.SetObject4BlobName(
            var_blob.logical_blob_name, var_blob.blob_object
        )
        return
    _NaiveInterpret(op_attribute, parallel_conf, blob_register)
    var_blob = _MakeEagerLogicalBlob(op_attribute, "out", blob_register=blob_register)
    EagerInitVariableBlob(op_attribute.op_conf, var_blob)
    sess.StashVariableBlob4Job(job_name, op_attribute.op_conf.name, var_blob)
    return var_blob


def _Watch(op_attribute, parallel_conf, blob_register):
    lbi = op_attribute.arg_signature.bn_in_op2lbi["in"]
    uuid = op_attribute.op_conf.foreign_watch_conf.handler_uuid
    lbn = "%s/%s" % (lbi.op_name, lbi.blob_name)
    in_blob_object = blob_register.GetObject4BlobName(lbn)
    if in_blob_object.op_arg_parallel_attr.is_mirrored():
        blob = remote_blob_util.EagerMirroredBlob(lbi, in_blob_object)
    else:
        blob = remote_blob_util.EagerConsistentBlob(lbi, in_blob_object)
    uuid2watch_handler = session_ctx.GetDefaultSession().uuid2watch_handler
    assert uuid in uuid2watch_handler
    uuid2watch_handler[uuid](blob)
    del uuid2watch_handler[uuid]


def _NaiveInterpret(op_attribute, parallel_conf, blob_register):
    def BuildInstruction(builder):
        with blob_register.BnInOp2BlobObjectScope(op_attribute) as bn_in_op2blob_object:
            builder.StatelessCall(
                op_attribute, parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object,
            )

    vm_util.LogicalRun(BuildInstruction)


def _MakeEagerLogicalBlob(op_attribute, obn, blob_register):
    lbi = op_attribute.arg_signature.bn_in_op2lbi[obn]
    blob_object = blob_register.GetObject4BlobName(
        "%s/%s" % (lbi.op_name, lbi.blob_name)
    )
    mirrored_sig_map = op_attribute.mirrored_signature.bn_in_op2opt_mirrored_parallel
    if mirrored_sig_map[obn].HasField("mirrored_parallel"):
        return remote_blob_util.EagerMirroredBlob(lbi, blob_object)
    else:
        return remote_blob_util.EagerConsistentBlob(lbi, blob_object)


def EagerInitVariableBlob(var_op_conf, var_blob):
    with oneflow.scope.placement("cpu", "0:0"):
        _Assign(var_blob.blob_object, _ModelInit(var_op_conf))


def _Assign(var_blob_object, value_blob_object):
    def BuildAssignInstruction(builder):
        new_parallel_desc_symbol = boxing_util.TryReplaceDeviceTag(
            builder, var_blob_object.parallel_desc_symbol, "cpu"
        )
        consumer_op_arg_parallel_attr = op_arg_util.OpArgParallelAttribute(
            new_parallel_desc_symbol,
            var_blob_object.op_arg_parallel_attr.sbp_parallel,
            var_blob_object.op_arg_parallel_attr.opt_mirrored_parallel,
        )
        tmp_blob_object = boxing_util.BoxingTo(
            builder, value_blob_object, consumer_op_arg_parallel_attr
        )
        boxing_util.Assign(builder, var_blob_object, tmp_blob_object)

    vm_util.LogicalRun(BuildAssignInstruction)


def _ModelInit(var_op_conf):
    op_conf, lbi = _GetModelInitAndLbi(var_op_conf)
    bn_in_op2blob_object = {}

    def BuildNotMirroredScope(old_scope, builder):
        return old_scope.BuildWithNewIsMirrored(builder, False)

    def BuildModeInitInstruction(builder):
        upstream_signature = op_attribute_pb.UpstreamSignature()
        parallel_conf = oneflow.placement.current_scope().default_parallel_conf
        op_conf.scope_symbol_id = oneflow.current_scope().symbol_id
        op_attribute = c_api_util.InferOpConf(op_conf, upstream_signature)
        builder.StatelessCall(
            op_attribute, parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object
        )

    sess = session_ctx.GetDefaultSession()
    with sess.NewCurrentScope(sess.MakeScope(BuildNotMirroredScope)):
        vm_util.LogicalRun(BuildModeInitInstruction)
    return bn_in_op2blob_object["out_0"]


def _GetModelInitAndLbi(var_op_conf):
    variable_op_conf = op_conf_util.VariableOpConf()
    variable_op_conf.CopyFrom(var_op_conf.variable_conf)
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = "model_init"
    op_conf.device_type = device_util.DeviceType4DeviceTag("cpu")
    op_conf.model_init_conf.out.append("out_0")
    op_conf.model_init_conf.variable_op_name.append(var_op_conf.name)
    op_conf.model_init_conf.original_variable_conf.append(variable_op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.model_init_conf.out[0]
    return op_conf, lbi

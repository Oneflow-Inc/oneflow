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

import oneflow.core.operator.op_node_signature_pb2 as op_node_signature_pb
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.operator.interface_blob_conf_pb2 as inter_face_blob_conf_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.boxing_util as boxing_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow.python.eager.symbol_storage as symbol_storage
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.experimental.name_scope as name_scope
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.eager.op_infer_util as op_infer_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow_api.oneflow.core.register.logical_blob_id as lbi_util
from google.protobuf import text_format

import oneflow
import oneflow_api
import numpy as np
import os

default_blob_register = blob_register_util.GetDefaultBlobRegister()


def Interpret(op_attribute, parallel_conf, blob_register):
    if op_attribute.op_conf.HasField("cast_to_mirrored_conf"):
        return MirroredCast(op_attribute, blob_register)
    if op_attribute.op_conf.HasField("cast_from_mirrored_conf"):
        return MirroredCast(op_attribute, blob_register)
    assert isinstance(parallel_conf, placement_cfg.ParallelConf)
    if op_attribute.op_conf.HasField("distribute_split_conf"):
        return DistributeSplitOrClone(op_attribute, parallel_conf, blob_register)
    if op_attribute.op_conf.HasField("distribute_clone_conf"):
        return DistributeSplitOrClone(op_attribute, parallel_conf, blob_register)
    if op_attribute.op_conf.HasField("distribute_concat_conf"):
        return DistributeConcatOrAdd(op_attribute, parallel_conf, blob_register)
    if op_attribute.op_conf.HasField("distribute_add_conf"):
        return DistributeConcatOrAdd(op_attribute, parallel_conf, blob_register)
    if op_attribute.op_conf.HasField("variable_conf"):
        return _FindOrCreateVarBlobObject(op_attribute, parallel_conf, blob_register)
    if op_attribute.op_conf.HasField("foreign_watch_conf"):
        return _Watch(op_attribute, parallel_conf, blob_register)
    return _NaiveInterpret(op_attribute, parallel_conf, blob_register)


def OpKernelCall(opkernel_object, op_attribute, blob_register):
    def BuildInstruction(builder):
        with blob_register_util.BnInOp2BlobObjectScope(
            blob_register, op_attribute
        ) as bn_in_op2blob_object:
            builder.StatefulCall(
                op_attribute,
                opkernel_object=opkernel_object,
                bn_in_op2blob_object=bn_in_op2blob_object,
            )

    vm_util.LogicalRun(BuildInstruction)


def MirroredCast(op_attribute, blob_register):
    def BuildInstruction(builder):
        with blob_register_util.BnInOp2BlobObjectScope(
            blob_register, op_attribute
        ) as bn_in_op2blob_object:
            in_blob_object = bn_in_op2blob_object["in"]
            parallel_desc_symbol = in_blob_object.parallel_desc_symbol
            op_arg_parallel_attr = oneflow_api.GetOpArgParallelAttribute(
                parallel_desc_symbol, str(op_attribute), "out"
            )
            out_blob_object = builder.MakeReferenceBlobObject(
                in_blob_object, op_arg_parallel_attr
            )
            bn_in_op2blob_object["out"] = out_blob_object

    vm_util.LogicalRun(BuildInstruction)


def DistributeSplitOrClone(op_attribute, parallel_conf, blob_register):
    parallel_sig = op_attribute.parallel_signature.bn_in_op2parallel_desc_symbol_id

    def GetInBlobObject(builder, ibn, bn_in_op2blob_object):
        origin_blob_object = bn_in_op2blob_object[ibn]
        in_op_parallel_desc_sym = oneflow_api.GetPlacementSymbol(parallel_sig[ibn])
        in_op_arg_parallel_attr = oneflow_api.GetOpArgParallelAttribute(
            in_op_parallel_desc_sym, str(op_attribute), ibn
        )
        return boxing_util.BoxingTo(
            builder, origin_blob_object, in_op_arg_parallel_attr
        )

    def BuildInstruction(builder):
        with blob_register_util.BnInOp2BlobObjectScope(
            blob_register, op_attribute
        ) as bn_in_op2blob_object:
            physical_out_blob_objects = builder.UnpackLogicalBlobToPhysicalBlobs(
                GetInBlobObject(builder, "in", bn_in_op2blob_object)
            )
            for i, blob_object in enumerate(physical_out_blob_objects):
                bn_in_op2blob_object["out_%s" % i] = blob_object

    vm_util.LogicalRun(BuildInstruction)


def DistributeConcatOrAdd(op_attribute, parallel_conf, blob_register):
    op_parallel_desc_sym = oneflow_api.GetPlacementSymbol(
        op_attribute.parallel_signature.op_parallel_desc_symbol_id
    )
    parallel_size = len(op_attribute.input_bns)
    op_arg_parallel_attr = oneflow_api.GetOpArgParallelAttribute(
        op_parallel_desc_sym, str(op_attribute), "out"
    )
    op_arg_blob_attr = oneflow_api.GetOpArgBlobAttribute(str(op_attribute), "out")
    parallel_sig = op_attribute.parallel_signature.bn_in_op2parallel_desc_symbol_id

    def GetInBlobObject(builder, i, bn_in_op2blob_object):
        ibn = "in_%s" % i
        origin_blob_object = bn_in_op2blob_object[ibn]
        in_op_parallel_desc_sym = oneflow_api.GetPlacementSymbol(parallel_sig[ibn])
        in_op_arg_parallel_attr = oneflow_api.GetOpArgParallelAttribute(
            in_op_parallel_desc_sym, str(op_attribute), ibn
        )
        return boxing_util.BoxingTo(
            builder, origin_blob_object, in_op_arg_parallel_attr
        )

    def BuildInstruction(builder):
        with blob_register_util.BnInOp2BlobObjectScope(
            blob_register, op_attribute
        ) as bn_in_op2blob_object:

            def GetPhysicalInBlob(i):
                return GetInBlobObject(builder, i, bn_in_op2blob_object)

            in_blob_objects = [GetPhysicalInBlob(i) for i in range(parallel_size)]
            bn_in_op2blob_object["out"] = builder.PackPhysicalBlobsToLogicalBlob(
                in_blob_objects, op_arg_parallel_attr, op_arg_blob_attr
            )

    vm_util.LogicalRun(BuildInstruction)


def _FindOrCreateVarBlobObject(op_attribute, parallel_conf, blob_register):
    job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
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
    EagerInitVariableBlob(sess, op_attribute.op_conf, var_blob)
    sess.StashVariableBlob4Job(job_name, op_attribute.op_conf.name, var_blob)
    return var_blob


def _Watch(op_attribute, parallel_conf, blob_register):
    lbi = op_attribute.arg_signature.bn_in_op2lbi["in"]
    uuid = op_attribute.op_conf.foreign_watch_conf.handler_uuid
    lbn = "%s/%s" % (lbi.op_name, lbi.blob_name)
    in_blob_object = blob_register.GetObject4BlobName(lbn)
    if not isinstance(lbi, lbi_util.LogicalBlobId):
        cfg_lbi = lbi_util.LogicalBlobId()
        cfg_lbi.set_op_name(lbi.op_name)
        cfg_lbi.set_blob_name(lbi.blob_name)
        lbi = cfg_lbi
    if in_blob_object.op_arg_parallel_attr.is_mirrored():
        blob = oneflow_api.EagerMirroredBlob(lbi, in_blob_object, default_blob_register)
    else:
        blob = oneflow_api.EagerConsistentBlob(
            lbi, in_blob_object, default_blob_register
        )
    uuid2watch_handler = session_ctx.GetDefaultSession().uuid2watch_handler
    assert uuid in uuid2watch_handler
    uuid2watch_handler[uuid](blob)
    del uuid2watch_handler[uuid]


def _NaiveInterpret(op_attribute, parallel_conf, blob_register):
    def BuildInstruction(builder):
        with blob_register_util.BnInOp2BlobObjectScope(
            blob_register, op_attribute
        ) as bn_in_op2blob_object:
            cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(
                str(op_attribute)
            )
            builder.StatelessCall(
                cfg_op_attribute,
                parallel_conf,
                bn_in_op2blob_object,
                boxing_util.BoxingTo,
                vm_util._FindOrCreateDelegateBlobObject,
            )

    vm_util.LogicalRun(BuildInstruction)


def _MakeEagerLogicalBlob(op_attribute, obn, blob_register):
    lbi = op_attribute.arg_signature.bn_in_op2lbi[obn]
    blob_object = blob_register.GetObject4BlobName(
        "%s/%s" % (lbi.op_name, lbi.blob_name)
    )
    mirrored_sig_map = op_attribute.mirrored_signature.bn_in_op2opt_mirrored_parallel
    if not isinstance(lbi, lbi_util.LogicalBlobId):
        cfg_lbi = lbi_util.LogicalBlobId()
        cfg_lbi.set_op_name(lbi.op_name)
        cfg_lbi.set_blob_name(lbi.blob_name)
        lbi = cfg_lbi
    if mirrored_sig_map[obn].HasField("mirrored_parallel"):
        return oneflow_api.EagerMirroredBlob(lbi, blob_object, default_blob_register)
    else:
        return oneflow_api.EagerConsistentBlob(lbi, blob_object, default_blob_register)


def EagerInitVariableBlob(sess, var_op_conf, var_blob):
    snapshot_path = sess.snapshot_mgr.get_snapshot_path(var_op_conf.name)
    with oneflow.scope.placement("cpu", "0:0"):
        if snapshot_path is None:
            blob_object = _EagerRunModelInit(var_op_conf)
        else:
            blob_object = _EagerRunModelLoad(var_op_conf, snapshot_path)

        _Assign(var_blob.blob_object, blob_object)


def EagerSaveVariableBlob(snapshot_path):
    var_blobs = session_ctx.GetDefaultSession().var_name2var_blob.values()
    with oneflow.scope.placement("cpu", "0:0"):
        _EagerRunModelSave(var_blobs, snapshot_path)


def _Assign(var_blob_object, value_blob_object):
    def BuildAssignInstruction(builder):
        new_parallel_desc_symbol = boxing_util.TryReplaceDeviceTag(
            builder, var_blob_object.parallel_desc_symbol, "cpu"
        )
        consumer_op_arg_parallel_attr = oneflow_api.OpArgParallelAttribute(
            new_parallel_desc_symbol,
            str(var_blob_object.op_arg_parallel_attr.sbp_parallel),
            str(var_blob_object.op_arg_parallel_attr.opt_mirrored_parallel),
        )
        tmp_blob_object = boxing_util.BoxingTo(
            builder, value_blob_object, consumer_op_arg_parallel_attr
        )
        boxing_util.Assign(builder, var_blob_object, tmp_blob_object)

    vm_util.LogicalRun(BuildAssignInstruction)


def _BuildNotMirroredScope(old_scope, builder):
    return builder.BuildScopeWithNewIsMirrored(old_scope, False)


def _EagerRunModelInit(var_op_conf):
    op_conf, _ = _GenModelInitOpConfAndRetLbi(var_op_conf)
    bn_in_op2blob_object = oneflow_api.deprecated.BnInOp2BlobObject()

    def BuildModelInitInstruction(builder):
        upstream_signature = op_node_signature_pb.OpNodeSignature()
        op_conf.scope_symbol_id = oneflow.current_scope().symbol_id
        op_attribute = c_api_util.InferOpConf(op_conf, upstream_signature)
        parallel_conf = (
            oneflow.current_scope().device_parallel_desc_symbol.parallel_conf
        )
        cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(
            str(op_attribute)
        )
        builder.StatelessCall(
            cfg_op_attribute,
            parallel_conf,
            bn_in_op2blob_object,
            boxing_util.BoxingTo,
            vm_util._FindOrCreateDelegateBlobObject,
        )

    sess = session_ctx.GetDefaultSession()
    with scope_util.ScopeContext(scope_util.MakeScope(_BuildNotMirroredScope)):
        vm_util.LogicalRun(BuildModelInitInstruction)

    return bn_in_op2blob_object["out_0"]


def _MakeModelIOPathInputBuilds(op_conf, path, bn_in_op2blob_object):
    def BuildModelIOPathInputInstruction(builder):
        op_attribute = op_infer_util.Infer(op_conf, ibn2blob_object={})
        parallel_conf = (
            oneflow.current_scope().device_parallel_desc_symbol.parallel_conf
        )
        cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(
            str(op_attribute)
        )
        builder.StatelessCall(
            cfg_op_attribute,
            parallel_conf,
            bn_in_op2blob_object,
            boxing_util.BoxingTo,
            vm_util._FindOrCreateDelegateBlobObject,
        )

    def FeedPath(ofblob):
        ofblob.CopyFromNdarray(np.frombuffer(path.encode("ascii"), dtype=np.int8))

    def BuildFeedPathInstruction(builder):
        blob_object = bn_in_op2blob_object["out"]
        builder.FeedBlob(blob_object, FeedPath)
        builder.InsertRemoveForeignCallbackInstruction(blob_object.object_id, FeedPath)

    return BuildModelIOPathInputInstruction, BuildFeedPathInstruction


def _EagerRunModelLoad(var_op_conf, snapshot_path):
    assert isinstance(snapshot_path, str)
    assert os.path.basename(snapshot_path) == "out"
    snapshot_path = os.path.dirname(snapshot_path)
    assert os.path.basename(snapshot_path) == var_op_conf.name
    snapshot_path = os.path.dirname(snapshot_path)

    path_input_op_conf, path_lbi = _GenModelIOPathInputOpConfAndRetLbi()
    path_input_blob_objects = {}
    (
        BuildModelIOPathInputInstruction,
        BuildFeedPathInstruction,
    ) = _MakeModelIOPathInputBuilds(
        path_input_op_conf, snapshot_path, path_input_blob_objects
    )

    model_load_op_conf, _ = _GenModelLoadOpConfAndRetLbi(var_op_conf, path_lbi)
    model_load_blob_objects = oneflow_api.deprecated.BnInOp2BlobObject()

    def BuildModelLoadInstruction(builder):
        path_blob_object = path_input_blob_objects["out"]
        model_load_blob_objects["path"] = path_blob_object
        op_attribute = op_infer_util.Infer(
            model_load_op_conf, ibn2blob_object=model_load_blob_objects
        )
        parallel_conf = path_blob_object.parallel_desc_symbol.parallel_conf
        cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(
            str(op_attribute)
        )
        builder.StatelessCall(
            cfg_op_attribute,
            parallel_conf,
            model_load_blob_objects,
            boxing_util.BoxingTo,
            vm_util._FindOrCreateDelegateBlobObject,
        )

    sess = session_ctx.GetDefaultSession()
    with scope_util.ScopeContext(scope_util.MakeScope(_BuildNotMirroredScope)):
        vm_util.LogicalRun(BuildModelIOPathInputInstruction)
        vm_util.LogicalRun(BuildFeedPathInstruction)
        vm_util.LogicalRun(BuildModelLoadInstruction)

    return model_load_blob_objects["out_0"]


def _EagerRunModelSave(var_blobs, snapshot_path):
    path_input_op_conf, path_lbi = _GenModelIOPathInputOpConfAndRetLbi()
    path_input_blob_objects = {}
    (
        BuildModelIOPathInputInstruction,
        BuildFeedPathInstruction,
    ) = _MakeModelIOPathInputBuilds(
        path_input_op_conf, snapshot_path, path_input_blob_objects
    )

    model_save_op_conf = _GenModelSaveOpConf(var_blobs, path_lbi)
    model_save_blob_objects = oneflow_api.deprecated.BnInOp2BlobObject()

    def BuildModelSaveInstruction(builder):
        path_blob_object = path_input_blob_objects["out"]
        model_save_blob_objects["path"] = path_blob_object
        for i, blob in enumerate(var_blobs):
            model_save_blob_objects["in_{}".format(i)] = blob.blob_object

        op_attribute = op_infer_util.Infer(
            model_save_op_conf, ibn2blob_object=model_save_blob_objects
        )
        parallel_conf = path_blob_object.parallel_desc_symbol.parallel_conf
        cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(
            str(op_attribute)
        )
        builder.StatelessCall(
            cfg_op_attribute,
            parallel_conf,
            model_save_blob_objects,
            boxing_util.BoxingTo,
            vm_util._FindOrCreateDelegateBlobObject,
        )

    sess = session_ctx.GetDefaultSession()
    with scope_util.ScopeContext(scope_util.MakeScope(_BuildNotMirroredScope)):
        vm_util.LogicalRun(BuildModelIOPathInputInstruction)
        vm_util.LogicalRun(BuildFeedPathInstruction)
        vm_util.LogicalRun(BuildModelSaveInstruction)


def _GenModelInitOpConfAndRetLbi(var_op_conf):
    variable_op_conf = op_conf_util.VariableOpConf()
    variable_op_conf.CopyFrom(var_op_conf.variable_conf)
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = "model_init"
    op_conf.device_tag = "cpu"
    op_conf.model_init_conf.out.append("out_0")
    op_conf.model_init_conf.variable_op_name.append(var_op_conf.name)
    op_conf.model_init_conf.original_variable_conf.append(variable_op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.model_init_conf.out[0]
    return op_conf, lbi


def _GenModelLoadOpConfAndRetLbi(var_op_conf, path_lbi):
    variable_op_conf = op_conf_util.VariableOpConf()
    variable_op_conf.CopyFrom(var_op_conf.variable_conf)

    op_conf = op_conf_util.OperatorConf()
    op_conf.name = "model_load"
    op_conf.device_tag = "cpu"
    op_conf.model_load_conf.path = "{}/{}".format(path_lbi.op_name, path_lbi.blob_name)
    op_conf.model_load_conf.out.append("out_0")
    op_conf.model_load_conf.variable_op_name.append(var_op_conf.name)
    op_conf.model_load_conf.original_variable_conf.append(variable_op_conf)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.model_load_conf.out[0]
    return op_conf, lbi


def _GenModelIOPathInputOpConfAndRetLbi():
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = "model_io_path_input"
    op_conf.device_tag = "cpu"
    op_conf.input_conf.out = "out"

    blob_conf = inter_face_blob_conf_util.InterfaceBlobConf()
    blob_conf.shape.dim.append(65536)
    blob_conf.data_type = oneflow_api.deprecated.GetProtoDtype4OfDtype(oneflow.int8)
    blob_conf.is_dynamic = True
    op_conf.input_conf.blob_conf.CopyFrom(blob_conf)

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.input_conf.out
    return op_conf, lbi


def _GenModelSaveOpConf(var_blobs, path_lbi):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = "model_save"
    op_conf.device_tag = "cpu"
    op_conf.model_save_conf.path = "{}/{}".format(path_lbi.op_name, path_lbi.blob_name)
    for blob in var_blobs:
        getattr(op_conf.model_save_conf, "in").append(blob.logical_blob_name)
        getattr(op_conf.model_save_conf, "key").append(blob.logical_blob_name)

    return op_conf

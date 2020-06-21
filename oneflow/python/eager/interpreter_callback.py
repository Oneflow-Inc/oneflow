from __future__ import absolute_import

import oneflow.python.eager.vm_util as vm_util

import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.gradient_util as gradient_util
import oneflow.core.operator.op_attribute_pb2 as op_attribute_pb
import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.python.framework.op_arg_util as op_arg_util
import oneflow.python.experimental.name_scope as name_scope
from google.protobuf import text_format
import oneflow.python.ops.get_variable as get_variable
import oneflow.python.eager.blob_register as blob_register_util


def MirroredCast(op_attribute_str, parallel_conf_str):
    op_attribute = text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())
    blob_register = blob_register_util.GetDefaultBlobRegister()
    is_cast_to_mirrored = op_attribute.op_conf.HasField("cast_to_mirrored_conf")
    is_cast_from_mirrored = op_attribute.op_conf.HasField("cast_from_mirrored_conf")
    assert is_cast_to_mirrored or is_cast_from_mirrored
    _MirroredCastAndAddOutputBlobReleaser(op_attribute, blob_register)
    bw_blob_register = gradient_util.GetDefaultBackwardBlobRegister()
    gradient_util.TrySetBackwardUsedBlobObject(
        op_attribute, blob_register, bw_blob_register
    )


def InterpretCompletedOp(op_attribute_str, parallel_conf_str):
    op_attribute = text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())
    blob_register = gradient_util.GetDefaultBackwardBlobRegister()
    _InterpretCompletedOp(op_attribute, parallel_conf_str, blob_register)
    gradient_util.ReleaseUnusedBlobObject(op_attribute, blob_register)


def _InterpretCompletedOp(op_attribute, parallel_conf_str, blob_register):
    if op_attribute.op_conf.HasField("cast_to_mirrored_conf"):
        return _MirroredCast(op_attribute, blob_register)
    if op_attribute.op_conf.HasField("cast_from_mirrored_conf"):
        return _MirroredCast(op_attribute, blob_register)
    parallel_conf = text_format.Parse(parallel_conf_str, placement_pb.ParallelConf())
    if op_attribute.op_conf.HasField("variable_conf"):
        return _FindOrCreateVarBlobObject(op_attribute, parallel_conf, blob_register)
    return _NaiveInterpret(op_attribute, parallel_conf, blob_register)


def _FindOrCreateVarBlobObject(op_attribute, parallel_conf, blob_register):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    name = name_scope.GetJobNameScopePrefix(job_name) + op_attribute.op_conf.name
    sess = session_ctx.GetDefaultSession()
    var_blob = sess.TryGetVariableBlobOfJobFromStash(job_name, name)
    if var_blob is not None:
        blob_register.SetObject4BlobName(var_blob.unique_name, var_blob.blob_object)
        return
    _NaiveInterpret(op_attribute, parallel_conf, blob_register)
    var_blob = _MakeEagerLogicalBlob(op_attribute, "out", blob_register=blob_register)
    get_variable.InitVariableBlob(op_attribute.op_conf, var_blob)
    sess.StashVariableBlob4Job(job_name, op_attribute.op_conf.name, var_blob)
    return var_blob


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


def _NaiveInterpret(op_attribute, parallel_conf, blob_register):
    def BuildInstruction(builder):
        with blob_register.BnInOp2BlobObjectScope(op_attribute) as bn_in_op2blob_object:
            builder.StatelessCall(
                op_attribute, parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object,
            )

    vm_util.LogicalRun(BuildInstruction)


def _MirroredCastAndAddOutputBlobReleaser(op_attribute, blob_register):
    _MirroredCast(op_attribute, blob_register)
    _AddOutputBlobObjectReleaser4InputBlobObject(op_attribute, blob_register)


def _MirroredCast(op_attribute, blob_register):
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


def _AddOutputBlobObjectReleaser4InputBlobObject(op_attribute, blob_register):
    in_lbi = op_attribute.arg_signature.bn_in_op2lbi["in"]
    in_lbn = "%s/%s" % (in_lbi.op_name, in_lbi.blob_name)
    in_blob_object = blob_register.GetObject4BlobName(in_lbn)
    release = _MakeReleaser4MirroredCastBlobObject(op_attribute, blob_register)
    in_blob_object.add_releaser(release)


def _MakeReleaser4MirroredCastBlobObject(op_attribute, blob_register):
    def ReleaseMirroredBlobObject(*args):
        for obn in op_attribute.output_bns:
            lbi = op_attribute.arg_signature.bn_in_op2lbi[obn]
            lbn = "%s/%s" % (lbi.op_name, lbi.blob_name)
            blob_object = blob_register.GetObject4BlobName(lbn)
            blob_register.ClearObject4BlobName(lbn)

    return ReleaseMirroredBlobObject

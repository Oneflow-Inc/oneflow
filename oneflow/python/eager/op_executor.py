from __future__ import absolute_import

import oneflow.python.eager.vm_util as vm_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.op_arg_util as op_arg_util
import oneflow.python.experimental.name_scope as name_scope
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.ops.get_variable as get_variable
import oneflow.core.job.placement_pb2 as placement_pb
from google.protobuf import text_format


def Interpret(op_attribute, parallel_conf_str, blob_register):
    if op_attribute.op_conf.HasField("cast_to_mirrored_conf"):
        return MirroredCast(op_attribute, blob_register)
    if op_attribute.op_conf.HasField("cast_from_mirrored_conf"):
        return MirroredCast(op_attribute, blob_register)
    parallel_conf = text_format.Parse(parallel_conf_str, placement_pb.ParallelConf())
    if op_attribute.op_conf.HasField("variable_conf"):
        return _FindOrCreateVarBlobObject(op_attribute, parallel_conf, blob_register)
    if op_attribute.op_conf.HasField("foreign_watch_conf"):
        return _Watch(op_attribute, parallel_conf, blob_register)
    return _NaiveInterpret(op_attribute, parallel_conf, blob_register)


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
    var_blob, job_var_blob = sess.TryGetVariableBlobOfJobFromStash(job_name, name)
    if var_blob is not None:
        blob_register.SetObject4BlobName(var_blob.unique_name, var_blob.blob_object)
        return
    _NaiveInterpret(op_attribute, parallel_conf, blob_register)
    var_blob = _MakeEagerLogicalBlob(op_attribute, "out", blob_register=blob_register)
    get_variable.InitVariableBlob(op_attribute.op_conf, var_blob)
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

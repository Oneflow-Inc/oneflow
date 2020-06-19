from __future__ import absolute_import

import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.object_cache as object_cache
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.core.operator.op_attribute_pb2 as op_attribute_pb
import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.python.framework.op_arg_util as op_arg_util
from google.protobuf import text_format


def Interpret(op_attribute_str, parallel_conf_str):
    op_attribute = text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())
    if op_attribute.op_conf.HasField("cast_to_mirrored_conf"):
        return _MirroredCast(op_attribute)
    if op_attribute.op_conf.HasField("cast_from_mirrored_conf"):
        return _MirroredCast(op_attribute)
    parallel_conf = text_format.Parse(parallel_conf_str, placement_pb.ParallelConf())
    return _Interpret(op_attribute, parallel_conf)


def BackwardInterpret(op_attribute_str, parallel_conf_str):
    op_attribute = text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())
    if op_attribute.op_conf.HasField("cast_to_mirrored_conf"):
        return _MirroredCast(op_attribute)
    if op_attribute.op_conf.HasField("cast_from_mirrored_conf"):
        return _MirroredCast(op_attribute)
    parallel_conf = text_format.Parse(parallel_conf_str, placement_pb.ParallelConf())
    return _Interpret(op_attribute, parallel_conf)


def _Interpret(op_attribute, parallel_conf):
    def BuildInstruction(builder):
        with object_cache.BnInOp2BlobObjectScope(op_attribute) as bn_in_op2blob_object:
            builder.StatelessCall(
                op_attribute, parallel_conf, bn_in_op2blob_object=bn_in_op2blob_object,
            )

    vm_util.LogicalRun(BuildInstruction)


def CastToMirrored(op_attribute_str, parallel_conf_str):
    op_attribute = text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())
    assert op_attribute.op_conf.HasField("cast_to_mirrored_conf")
    return _MirroredCast(op_attribute)


def CastFromMirrored(op_attribute_str, parallel_conf_str):
    op_attribute = text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())
    assert op_attribute.op_conf.HasField("cast_from_mirrored_conf")
    return _MirroredCast(op_attribute)


def _MirroredCast(op_attribute):
    def BuildInstruction(builder):
        with object_cache.BnInOp2BlobObjectScope(op_attribute) as bn_in_op2blob_object:
            in_blob_object = bn_in_op2blob_object["in"]
            parallel_desc_symbol = in_blob_object.parallel_desc_symbol
            op_arg_attribute = op_arg_util.GetOpArgAttribute(
                parallel_desc_symbol, op_attribute, "out"
            )
            out_blob_object = builder.MakeReferenceBlobObject(
                in_blob_object, op_arg_attribute
            )
            bn_in_op2blob_object["out"] = out_blob_object
        in_blob_object.add_releaser(_MakeReleaser4MirroredCastBlobObject(op_attribute))

    vm_util.LogicalRun(BuildInstruction)


def _MakeReleaser4MirroredCastBlobObject(op_attribute):
    def ReleaseMirroredBlobObject(*args):
        for obn in op_attribute.output_bns:
            lbi = op_attribute.arg_signature.bn_in_op2lbi[obn]
            lbn = "%s/%s" % (lbi.op_name, lbi.blob_name)
            blob_object = object_cache.GetObject4BlobName(lbn)
            blob_cache_util.TryDisableBlobCache(blob_object)
            object_cache.ClearObject4BlobName(lbn)

    return ReleaseMirroredBlobObject

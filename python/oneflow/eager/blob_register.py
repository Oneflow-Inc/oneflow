import oneflow
import oneflow._oneflow_internal
from contextlib import contextmanager


@contextmanager
def BnInOp2BlobObjectScope(blob_register, op_attribute):
    bn_in_op2blob_object = oneflow._oneflow_internal.deprecated.BnInOp2BlobObject()
    for ibn in op_attribute.input_bns:
        lbi = op_attribute.arg_signature.bn_in_op2lbi[ibn]
        bn_in_op2blob_object[ibn] = blob_register.GetObject4BlobName(
            "%s/%s" % (lbi.op_name, lbi.blob_name)
        )
    yield bn_in_op2blob_object
    for obn in op_attribute.output_bns:
        lbi = op_attribute.arg_signature.bn_in_op2lbi[obn]
        blob_register.SetObject4BlobName(
            "%s/%s" % (lbi.op_name, lbi.blob_name), bn_in_op2blob_object[obn]
        )

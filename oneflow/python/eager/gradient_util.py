from __future__ import absolute_import

import oneflow.python.framework.session_context as session_ctx


def GetDefaultBackwardBlobRegister():
    return session_ctx.GetDefaultSession().backward_blob_register


def ReleaseUnusedBlobObject(op_attribute, blob_register):
    assert op_attribute.HasField("blob_last_used_signature")
    signature_map = op_attribute.blob_last_used_signature.bn_in_op2blob_last_used
    bn_in_op2lbi = op_attribute.arg_signature.bn_in_op2lbi
    for bn_in_op, is_blob_last_used in signature_map.items():
        if not is_blob_last_used:
            continue
        lbi = bn_in_op2lbi[bn_in_op]
        blob_register.ClearObject4BlobName("%s/%s" % (lbi.op_name, lbi.blob_name))

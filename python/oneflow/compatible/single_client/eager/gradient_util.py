from oneflow.compatible.single_client.python.framework import session_context as session_ctx

def GetDefaultBackwardBlobRegister():
    return session_ctx.GetDefaultSession().backward_blob_register

def ReleaseUnusedBlobObject(op_attribute, blob_register):
    assert op_attribute.HasField('blob_last_used_signature'), op_attribute
    signature_map = op_attribute.blob_last_used_signature.bn_in_op2blob_last_used
    bn_in_op2lbi = op_attribute.arg_signature.bn_in_op2lbi
    for (bn_in_op, is_blob_last_used) in signature_map.items():
        if not is_blob_last_used:
            continue
        lbi = bn_in_op2lbi[bn_in_op]
        lbn = '%s/%s' % (lbi.op_name, lbi.blob_name)
        blob_register.ClearObject4BlobName(lbn)

def TrySetBackwardUsedBlobObject(op_attribute, fw_blob_register, bw_blob_register):
    assert op_attribute.HasField('blob_backward_used_signature'), op_attribute
    signature_map = op_attribute.blob_backward_used_signature.bn_in_op2blob_backward_used
    bn_in_op2lbi = op_attribute.arg_signature.bn_in_op2lbi
    for (bn_in_op, is_blob_backward_used) in signature_map.items():
        if not is_blob_backward_used:
            continue
        lbi = bn_in_op2lbi[bn_in_op]
        lbn = '%s/%s' % (lbi.op_name, lbi.blob_name)
        blob_object = fw_blob_register.GetObject4BlobName(lbn)
        bw_blob_register.TrySetObject4BlobName(lbn, blob_object)
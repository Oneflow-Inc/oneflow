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

import oneflow.python.framework.session_context as session_ctx


def GetDefaultBackwardBlobRegister():
    return session_ctx.GetDefaultSession().backward_blob_register


def ReleaseUnusedBlobObject(op_attribute, blob_register):
    assert op_attribute.HasField("blob_last_used_signature"), op_attribute
    signature_map = op_attribute.blob_last_used_signature.bn_in_op2blob_last_used
    bn_in_op2lbi = op_attribute.arg_signature.bn_in_op2lbi
    for bn_in_op, is_blob_last_used in signature_map.items():
        if not is_blob_last_used:
            continue
        lbi = bn_in_op2lbi[bn_in_op]
        lbn = "%s/%s" % (lbi.op_name, lbi.blob_name)
        blob_register.ClearObject4BlobName(lbn)


def TrySetBackwardUsedBlobObject(op_attribute, fw_blob_register, bw_blob_register):
    assert op_attribute.HasField("blob_backward_used_signature"), op_attribute
    signature_map = (
        op_attribute.blob_backward_used_signature.bn_in_op2blob_backward_used
    )
    bn_in_op2lbi = op_attribute.arg_signature.bn_in_op2lbi
    for bn_in_op, is_blob_backward_used in signature_map.items():
        if not is_blob_backward_used:
            continue
        lbi = bn_in_op2lbi[bn_in_op]
        lbn = "%s/%s" % (lbi.op_name, lbi.blob_name)
        blob_object = fw_blob_register.GetObject4BlobName(lbn)
        bw_blob_register.TrySetObject4BlobName(lbn, blob_object)

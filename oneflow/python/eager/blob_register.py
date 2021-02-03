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

import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow_api
from contextlib import contextmanager


def GetDefaultBlobRegister():
    return default_blob_register_


@contextmanager
def BnInOp2BlobObjectScope(blob_register, op_attribute):
    bn_in_op2blob_object = oneflow_api.deprecated.BnInOp2BlobObject()
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


default_blob_register_ = oneflow_api.BlobRegister(blob_cache_util.TryDisableBlobCache)

from __future__ import absolute_import

import oneflow.python.eager.blob_cache as blob_cache_util
from contextlib import contextmanager


def GetDefaultBlobRegister():
    return default_blob_register_


class BlobRegister(object):
    def __init__(self):
        self.blob_name2object_ = {}

    @property
    def blob_name2object(self):
        return self.blob_name2object_

    def HasObject4BlobName(self, blob_name):
        return blob_name in self.blob_name2object

    def GetObject4BlobName(self, blob_name):
        assert self.HasObject4BlobName(blob_name), "blob_name %s not found" % blob_name
        return self.blob_name2object[blob_name]

    def SetObject4BlobName(self, blob_name, obj):
        assert not self.HasObject4BlobName(blob_name), blob_name
        self.blob_name2object[blob_name] = obj

    def TrySetObject4BlobName(self, blob_name, obj):
        if not self.HasObject4BlobName(blob_name):
            self.SetObject4BlobName(blob_name, obj)

    def ClearObject4BlobName(self, blob_name):
        assert self.HasObject4BlobName(blob_name), "blob_name %s not found" % blob_name
        blob_cache_util.TryDisableBlobCache(self.blob_name2object[blob_name])
        del self.blob_name2object[blob_name]

    def TryClearObject4BlobName(self, blob_name):
        if self.HasObject4BlobName(blob_name):
            self.ClearObject4BlobName(blob_name)

    @contextmanager
    def BnInOp2BlobObjectScope(self, op_attribute):
        bn_in_op2blob_object = {}
        for ibn in op_attribute.input_bns:
            lbi = op_attribute.arg_signature.bn_in_op2lbi[ibn]
            bn_in_op2blob_object[ibn] = self.GetObject4BlobName(
                "%s/%s" % (lbi.op_name, lbi.blob_name)
            )
        yield bn_in_op2blob_object
        for obn in op_attribute.output_bns:
            lbi = op_attribute.arg_signature.bn_in_op2lbi[obn]
            self.SetObject4BlobName(
                "%s/%s" % (lbi.op_name, lbi.blob_name), bn_in_op2blob_object[obn]
            )


default_blob_register_ = BlobRegister()

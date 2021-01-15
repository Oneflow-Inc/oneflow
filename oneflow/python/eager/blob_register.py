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
import oneflow.python.framework.python_interpreter_util as python_interpreter_util
from contextlib import contextmanager


def GetDefaultBlobRegister():
    return default_blob_register_


class RegisteredBlobAccess(object):
    def __init__(self, blob_name, blob_register, blob_object=None):
        self.blob_name_ = blob_name
        self.blob_register_ = blob_register
        if blob_object is None:
            blob_object = blob_register.GetObject4BlobName(blob_name)
        else:
            blob_register.SetObject4BlobName(blob_name, blob_object)
        self.blob_object_ = blob_object
        self.reference_counter_ = 0

    @property
    def reference_counter(self):
        return self.reference_counter_

    def increase_reference_counter(self):
        self.reference_counter_ = self.reference_counter_ + 1

    def decrease_reference_counter(self):
        self.reference_counter_ = self.reference_counter_ - 1
        return self.reference_counter_

    @property
    def blob_object(self):
        return self.blob_object_

    def __del__(self):
        self.blob_register_.ClearObject4BlobName(self.blob_name_)


class BlobRegister(object):
    def __init__(self):
        self.blob_name2object_ = {}
        self.blob_name2access_ = {}

    def OpenRegisteredBlobAccess(self, blob_name, blob_object=None):
        if blob_name not in self.blob_name2access_:
            self.blob_name2access_[blob_name] = RegisteredBlobAccess(
                blob_name, self, blob_object
            )
        access = self.blob_name2access_[blob_name]
        access.increase_reference_counter()
        return access

    def CloseRegisteredBlobAccess(self, blob_name):
        if blob_name in self.blob_name2access_:
            if self.blob_name2access_[blob_name].decrease_reference_counter() == 0:
                del self.blob_name2access_[blob_name]

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

    def ForceReleaseAll(self, is_shutting_down=python_interpreter_util.IsShuttingDown):
        # Bind `python_interpreter_util.IsShuttingDown` early.
        # See the comments of `python_interpreter_util.IsShuttingDown`
        for blob_name, blob_object in self.blob_name2object.items():
            if is_shutting_down():
                return
            print("Forcely release blob %s." % blob_name)
            blob_object.ForceReleaseAll()

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

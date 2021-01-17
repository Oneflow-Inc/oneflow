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
import oneflow.python.framework.python_interpreter_util as python_interpreter_util


def FindOrCreateBlobCache(blob_object):
    object_id = blob_object.object_id
    global object_id2blob_cache
    if object_id not in object_id2blob_cache:
        object_id2blob_cache[object_id] = BlobCache(blob_object)
    return object_id2blob_cache[object_id]


def TryDisableBlobCache(blob_object):
    global object_id2blob_cache
    if blob_object.object_id not in object_id2blob_cache:
        return
    del object_id2blob_cache[blob_object.object_id]


class BlobCache(object):
    def __init__(self, blob_object):
        self.blob_object_ = blob_object
        self.header_cache_ = None
        self.body_cache_ = None
        self.delegate_blob_object_ = {}
        self.numpy_mirrored_list_ = None
        self.numpy_ = None

    @property
    def blob_object(self):
        return self.blob_object_

    def GetHeaderCache(self, fetch):
        if self.header_cache_ is None:
            self.header_cache_ = fetch(self.blob_object_)
        return self.header_cache_

    def GetBodyCache(self, fetch):
        if self.body_cache_ is None:
            self.body_cache_ = fetch(self.blob_object_)
        return self.body_cache_

    def GetCachedDelegateBlobObject(self, op_arg_parallel_attr, fetch):
        if op_arg_parallel_attr not in self.delegate_blob_object_:
            delegate_blob_object = fetch(self.blob_object, op_arg_parallel_attr)
            self.delegate_blob_object_[op_arg_parallel_attr] = delegate_blob_object
        return self.delegate_blob_object_[op_arg_parallel_attr]

    def GetCachedNumpyMirroredList(self, fetch):
        if self.numpy_mirrored_list_ is None:
            self.numpy_mirrored_list_ = fetch(self.blob_object_)
        return self.numpy_mirrored_list_

    def GetCachedNumpy(self, fetch):
        if self.numpy_ is None:
            self.numpy_ = fetch(self.blob_object_)
        return self.numpy_

    def __del__(self, is_shutting_down=python_interpreter_util.IsShuttingDown):
        # Bind `python_interpreter_util.IsShuttingDown` early.
        # See the comments of `python_interpreter_util.IsShuttingDown`
        for key in list(self.delegate_blob_object_.keys()):
            if is_shutting_down():
                return
            if self.delegate_blob_object_[key] is not None:
                del self.delegate_blob_object_[key]


object_id2blob_cache = {}

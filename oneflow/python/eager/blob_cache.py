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

import oneflow
import oneflow_api


def GetBodyCache(self, fetch):
    if self.body_cache_ is None:
        self.body_cache_ = fetch(self.blob_object)
    return self.body_cache_


def GetCachedNumpyMirroredList(self, fetch):
    if self.numpy_mirrored_list_ is None:
        self.numpy_mirrored_list_ = fetch(self.blob_object)
    return self.numpy_mirrored_list_


def GetCachedNumpy(self, fetch):
    if self.numpy_ is None:
        self.numpy_ = fetch(self.blob_object)
    return self.numpy_


def RegisterMethodAndAttr4BlobCache():
    oneflow_api.BlobCache.body_cache_ = None
    oneflow_api.BlobCache.numpy_mirrored_list_ = None
    oneflow_api.BlobCache.numpy_ = None
    oneflow_api.BlobCache.GetBodyCache = GetBodyCache
    oneflow_api.BlobCache.GetCachedNumpyMirroredList = GetCachedNumpyMirroredList
    oneflow_api.BlobCache.GetCachedNumpy = GetCachedNumpy

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
    if not hasattr(self, "body_cache_"):
        self.body_cache_ = fetch(self.blob_object)
    return self.body_cache_


def GetCachedNumpy(self, fetch):
    if not hasattr(self, "numpy_"):
        self.numpy_ = fetch(self.blob_object)
    return self.numpy_


def RegisterMethodAndAttr4BlobCache():
    # BlobCache has will be registered three attr in these fun: body_cache_,  numpy_
    oneflow_api.BlobCache.GetBodyCache = GetBodyCache
    oneflow_api.BlobCache.GetCachedNumpy = GetCachedNumpy

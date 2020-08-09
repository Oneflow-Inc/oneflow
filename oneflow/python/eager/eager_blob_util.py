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
import oneflow.python.eager.blob_register as blob_register_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.framework.blob_trait as blob_trait
import oneflow.python.framework.python_callback as python_callback
import oneflow.python.lib.core.async_util as async_util

blob_register = blob_register_util.GetDefaultBlobRegister()


class EagerPhysicalBlob(blob_trait.BlobOperatorTrait, blob_trait.BlobHeaderTrait):
    def __init__(self, blob_name):
        self.blob_name_ = blob_name
        self.blob_object_ = blob_register.GetObject4BlobName(blob_name)

    @property
    def logical_blob_name(self):
        return self.blob_name_

    @property
    def unique_name(self):
        return self.blob_name_

    @property
    def static_shape(self):
        return _GetPhysicalBlobHeaderCache(self.blob_object_).static_shape

    @property
    def shape(self):
        return _GetPhysicalBlobHeaderCache(self.blob_object_).shape

    @property
    def dtype(self):
        return _GetPhysicalBlobHeaderCache(self.blob_object_).dtype

    @property
    def is_dynamic(self):
        return True

    @property
    def is_tensor_list(self):
        return _GetPhysicalBlobHeaderCache(self.blob_object_).is_tensor_list

    def numpy(self):
        assert not self.is_tensor_list
        return _GetPhysicalBlobBodyCache(self.blob_object_)

    def numpy_list(self):
        assert self.is_tensor_list
        return _GetPhysicalBlobBodyCache(self.blob_object_)

    def __str__(self):
        return "EagerPhysicalBlob(shape=%s, dtype=%s, is_tensor_list=%s)" % (
            self.shape,
            self.dtype,
            self.is_tensor_list,
        )

    def __del__(self):
        blob_register.ClearObject4BlobName(self.unique_name)


def FetchTensorBlobAsNumpyList(parallel_size, blob_object):
    def AsyncFetchBlobBody(Yield):
        fetcher = _MakeFetcherEagerBlobBodyAsNumpyFromOfBlob(Yield)

        def BuildFetchBlobBodyInstruction(builder):
            builder.FetchBlobBody(blob_object, fetcher)
            builder.InsertRemoveForeignCallbackInstruction(
                blob_object.object_id, fetcher
            )

        vm_util.PhysicalRun(BuildFetchBlobBodyInstruction)

    return async_util.Await(parallel_size, AsyncFetchBlobBody)


def _GetPhysicalBlobHeaderCache(blob_object):
    blob_cache = blob_cache_util.FindOrCreateBlobCache(blob_object)
    return blob_cache.GetHeaderCache(_FetchBlobHeader)


def _GetPhysicalBlobBodyCache(blob_object):
    blob_cache = blob_cache_util.FindOrCreateBlobCache(blob_object)
    return blob_cache.GetBodyCache(_FetchPhysicalBlobBody)


def _FetchBlobHeader(blob_object):
    def AsyncFetchBlobHeader(Yield):
        fetcher = _MakeFetcherEagerPhysicalBlobHeaderFromOfBlob(Yield)

        def BuildFetchBlobHeaderInstruction(builder):
            builder.FetchBlobHeader(blob_object, fetcher)
            builder.InsertRemoveForeignCallbackInstruction(
                blob_object.object_id, fetcher
            )

        vm_util.PhysicalRun(BuildFetchBlobHeaderInstruction)

    return async_util.Await(1, AsyncFetchBlobHeader)[0]


def _FetchPhysicalBlobBody(blob_object):
    return FetchTensorBlobAsNumpyList(1, blob_object)[0]


def _MakeFetcherEagerPhysicalBlobHeaderFromOfBlob(Yield):
    def Callback(ofblob):
        Yield(
            EagerPhysicalBlobHeader(
                ofblob.static_shape,
                ofblob.shape_list,
                ofblob.dtype,
                ofblob.is_tensor_list,
            )
        )

    return Callback


def _MakeFetcherEagerBlobBodyAsNumpyFromOfBlob(Yield):
    def FetchFromOfBlob(ofblob):
        if ofblob.is_tensor_list:
            Yield(ofblob.CopyToFlatNdarrayList())
        else:
            Yield(ofblob.CopyToNdarray())

    return FetchFromOfBlob


class EagerPhysicalBlobHeader(object):
    def __init__(self, static_shape, shape_list, dtype, is_tensor_list):
        self.static_shape_ = static_shape
        self.shape_list_ = shape_list
        self.dtype_ = dtype
        self.is_tensor_list_ = is_tensor_list

    @property
    def static_shape(self):
        return self.static_shape_

    @property
    def shape(self):
        assert len(self.shape_list_) == 1
        assert not self.is_tensor_list_
        return self.shape_list_[0]

    @property
    def shape_list(self):
        assert self.is_tensor_list_
        return self.shape_list_

    @property
    def dtype(self):
        return self.dtype_

    @property
    def is_tensor_list(self):
        return self.is_tensor_list_

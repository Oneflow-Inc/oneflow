from __future__ import absolute_import

import oneflow.python.framework.blob_trait as blob_trait
import oneflow.python.eager.object_cache as object_cache
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.physical_blob_callback as physical_blob_callback
import oneflow.python.lib.core.async_util as async_util


class EagerPhysicalBlob(blob_trait.BlobOperatorTrait, blob_trait.BlobHeaderTrait):
    def __init__(self, blob_name):
        self.blob_name_ = blob_name
        self.blob_object_ = object_cache.GetObject4BlobName(blob_name)

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

    def __str__(self):
        return "EagerPhysicalBlob(shape=%s, dtype=%s, is_tensor_list=%s)" % (
            self.shape,
            self.dtype,
            self.is_tensor_list,
        )

    def __del__(self):
        blob_cache_util.TryDisableBlobCache(self.blob_object_)
        object_cache.ClearObject4BlobName(self.unique_name)


def FetchTensorBlobAsNumpyList(parallel_size, blob_object):
    def AsyncFetchBlobBody(Yield):
        fetcher = _MakeFetcherEagerBlobBodyAsNumpyFromOfBlob(Yield)
        vm_util.PhysicalRun(lambda builder: builder.WatchBlobBody(blob_object, fetcher))
        physical_blob_callback.DeleteRegisteredCallback(fetcher)

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
        vm_util.PhysicalRun(
            lambda builder: builder.WatchBlobHeader(blob_object, fetcher)
        )
        physical_blob_callback.DeleteRegisteredCallback(fetcher)

    return async_util.Await(1, AsyncFetchBlobHeader)[0]


def _FetchPhysicalBlobBody(blob_object):
    return FetchTensorBlobAsNumpyList(1, blob_object)[0]


def _MakeFetcherEagerPhysicalBlobHeaderFromOfBlob(Yield):
    def Callback(ofblob):
        # TODO(lixinqi) refactor ofblob.static_shape ofblob.shape_list
        static_shape = ofblob.static_shape
        shape = ofblob.shape
        Yield(
            EagerPhysicalBlobHeader(shape, [shape], ofblob.dtype, ofblob.is_tensor_list)
        )

    return Callback


def _MakeFetcherEagerBlobBodyAsNumpyFromOfBlob(Yield):
    return lambda ofblob: Yield(ofblob.CopyToNdarray())


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
    def dtype(self):
        return self.dtype_

    @property
    def is_tensor_list(self):
        return self.is_tensor_list_

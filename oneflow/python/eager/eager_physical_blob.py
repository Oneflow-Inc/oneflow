from __future__ import absolute_import

import oneflow.python.framework.blob_trait as blob_trait
import oneflow.python.eager.object_dict as object_dict
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.lib.core.async_util as async_util

class EagerPhysicalBlob(blob_trait.BlobOperatorTrait, blob_trait.BlobHeaderTrait):
    def __init__(self, blob_name):
        self.blob_name_ = blob_name
        self.blob_object = object_dict.GetObject4BlobName(blob_name)

    @property
    def logical_blob_name(self): return self.blob_name_

    @property
    def static_shape(self):
        return _GetBlobHeaderCache(self.blob_object).static_shape

    @property
    def shape(self):
        return _GetBlobHeaderCache(self.blob_object).shape

    @property
    def dtype(self):
        return _GetBlobHeaderCache(self.blob_object).dtype

    @property
    def is_dynamic(self): return True

    @property
    def is_tensor_list(self):
        return _GetBlobHeaderCache(self.blob_object).is_tensor_list

    def numpy(self):
        assert not self.is_tensor_list
        return _GetBlobBodyCache(self.blob_object)

    def __str__(self):
        return ("EagerPhysicalBlob(shape=%s, dtype=%s, is_tensor_list=%s)"
                %(self.shape, self.dtype, self.is_tensor_list))

def _GetBlobHeaderCache(blob_object):
    blob_cache = blob_cache_util.FindOrCreateBlobCache(blob_object)
    return blob_cache.GetHeaderCache(_FetchBlobHeader)

def _GetBlobBodyCache(blob_object):
    blob_cache = blob_cache_util.FindOrCreateBlobCache(blob_object)
    return blob_cache.GetBodyCache(_FetchBlobBody)

def _FetchBlobHeader(blob_object):
    def AsyncFetchBlobHeader(Return):
        fetcher = MakeFetherEagerPhysicalBlobHeaderFromOfBlob(Return)
        vm_util.PhysicalRun(lambda builder: builder.WatchBlobHeader(blob_object, fetcher))
    return async_util.Async2Sync(1, AsyncFetchBlobHeader)[0]

def _FetchBlobBody(blob_object):
    def AsyncFetchBlobBody(Return):
        fetcher = MakeFetherEagerPhysicalBlobBodyFromOfBlob(Return)
        vm_util.PhysicalRun(lambda builder: builder.WatchBlobBody(blob_object, fetcher))
    return async_util.Async2Sync(1, AsyncFetchBlobBody)[0]

def MakeFetherEagerPhysicalBlobHeaderFromOfBlob(Return):
    def Callback(ofblob):
        # TODO(lixinqi) refactor ofblob.static_shape ofblob.shape_list
        shape = ofblob.shape
        Return(EagerPhysicalBlobHeader(shape, [shape], ofblob.dtype, ofblob.is_tensor_list))
    return Callback


def MakeFetherEagerPhysicalBlobBodyFromOfBlob(Return):
    return lambda ofblob: Return(ofblob.CopyToNdarray())

    
class EagerPhysicalBlobHeader(object):
    def __init__(self, static_shape, shape_list, dtype, is_tensor_list):
        self.static_shape_ = static_shape
        self.shape_list_ = shape_list
        self.dtype_ = dtype
        self.is_tensor_list_ = is_tensor_list

    @property
    def static_shape(self): return self.static_shape_

    @property
    def shape(self):
        assert len(self.shape_list_) == 1 
        assert not self.is_tensor_list_
        return self.shape_list_[0]

    @property
    def dtype(self): return self.dtype_

    @property
    def is_tensor_list(self): return self.is_tensor_list_

from __future__ import absolute_import

import oneflow.python.framework.blob_trait as blob_trait
import oneflow.python.eager.object_dict as object_dict

class EagerPhysicalBlob(blob_trait.BlobOperatorTrait, blob_trait.BlobHeaderTrait):
    def __init__(self, blob_name):
        self.blob_name_ = blob_name
        self.blob_object_id_ = id_cache.GetObjectId4BlobName(blob_name)

    @property
    def static_shape(self):
        return _GetBlobHeaderCache(self.blob_object_id_).static_shape

    @property
    def shape(self):
        return _GetBlobHeaderCache(self.blob_object_id_).shape

    @property
    def dtype(self):
        return _GetBlobHeaderCache(self.blob_object_id_).dtype

    @property
    def is_dynamic(self): return True

    @property
    def is_tensor_list(self):
        return _GetBlobHeaderCache(self.blob_object_id_).is_tensor_list

def _GetBlobHeaderCache(blob_object_id):
    TODO()

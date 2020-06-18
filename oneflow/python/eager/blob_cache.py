from __future__ import absolute_import


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

    def GetCachedDelegateBlobObject(self, op_arg_attribute, fetch):
        if op_arg_attribute not in self.delegate_blob_object_:
            delegate_blob_object = fetch(self.blob_object, op_arg_attribute)
            self.delegate_blob_object_[op_arg_attribute] = delegate_blob_object
        return self.delegate_blob_object_[op_arg_attribute]

    def GetCachedNumpyMirroredList(self, fetch):
        if self.numpy_mirrored_list_ is None:
            self.numpy_mirrored_list_ = fetch(self.blob_object_)
        return self.numpy_mirrored_list_


object_id2blob_cache = {}

from __future__ import absolute_import

def FindOrCreateBlobCache(blob_object):
    object_id = blob_object.object_id
    global object_id2blob_cache
    if object_id not in object_id2blob_cache:
        object_id2blob_cache[object_id] = BlobCache(blob_object)
    return object_id2blob_cache[object_id]

def TryDisableBlobCache(blob_object):
    global object_id2blob_cache
    if blob_object.object_id not in object_id2blob_cache: return
    del object_id2blob_cache[blob_object.object_id]

class BlobCache(object):
    def __init__(self, blob_object):
        self.blob_object_ = blob_object
        self.header_cache_ = None
        self.body_cache_ = None
        self.delegate_blob_cache_ = {}

    @property
    def blob_object(self): return self.blob_object_

    def GetHeaderCache(self, fetch):
        if self.header_cache_ is None: self.header_cache_ = fetch(self.blob_object_)
        return self.header_cache_

    def GetBodyCache(self, fetch):
        if self.body_cache_ is None: self.body_cache_ = fetch(self.blob_object_)
        return self.body_cache_

    def GetCachedDelegateBlobObject(self, parallel_desc_symbol, fetch, release):
        if id(parallel_desc_symbol) not in self.delegate_blob_cache_:
            delegate_blob_object = fetch(self.blob_object, parallel_desc_symbol)
            self.delegate_blob_cache_[id(parallel_desc_symbol)] = RAIIBlobObject(
                    delegate_blob_object, release)
        return self.delegate_blob_cache_[id(parallel_desc_symbol)].blob_object


class RAIIBlobObject(object):
    def __init__(self, blob_object, release):
        self.blob_object_ = blob_object
        self.release_ = release

    @property
    def blob_object(self): return self.blob_object_

    def __del__(self):
        self.release_(self.blob_object_)

object_id2blob_cache = {}

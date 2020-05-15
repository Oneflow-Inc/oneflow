from __future__ import absolute_import

import oneflow.python.eager.object_parallel as object_parallel_util

def FindOrCreateBlobCache(object_id):
    global object_id2blob_cache
    if object_id not in object_id2blob_cache:
        object_id2blob_cache[object_id] = BlobCache(object_id)
    return object_id2blob_cache[object_id]

def DisableBlobCache(object_id):
    global object_id2blob_cache
    del object_id2blob_cache[object_id]

class BlobCache(object):
    def __init__(self, object_id):
        self.object_id_ = object_id
        self.header_cache_ = None
        self.body_cache_ = None
        # delegate object_id in another parallel_conf 
        self.delegate_object_ids_cache_ = None

    @property
    def object_id(self): return self.object_id_

    def GetHeaderCache(self, fetch):
        if self.header_cache_ is None: self.header_cache_ = fetch(self.object_id_)
        return self.header_cache_

    def GetBodyCache(self, fetch):
        if self.body_cache_ is None: self.body_cache_ = fetch(self.object_id_)
        return self.body_cache_

object_id2blob_cache = {}

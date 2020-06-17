from __future__ import absolute_import

import oneflow.python.eager.symbol as symbol_util


class Object(object):
    def __init__(self, object_id, parallel_desc_symbol):
        self.object_id_ = object_id
        self.parallel_desc_symbol_ = parallel_desc_symbol

    @property
    def object_id(self):
        return self.object_id_

    @property
    def parallel_desc_symbol(self):
        return self.parallel_desc_symbol_


class BlobObject(Object):
    def __init__(self, object_id, parallel_desc_symbol, release):
        Object.__init__(self, object_id, parallel_desc_symbol)
        self.release_ = [release]

    def add_releaser(self, release):
        self.release_.append(release)

    def __del__(self):
        for release in self.release_:
            release(self)
        self.release_ = []

from __future__ import absolute_import

import oneflow.core.common.data_type_pb2 as data_type_util

class BlobDesc(object):
    def __init__(self):
        pass

    @property
    def shape(self): return self.static_shape
    
    @property
    def static_shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def has_batch_dim(self):
        raise NotImplementedError

    @property
    def is_dynamic(self):
        raise NotImplementedError

    @property
    def split_axis(self):
        raise NotImplementedError

    @property
    def broadcast(self):
        raise NotImplementedError

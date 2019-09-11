from __future__ import absolute_import

import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.python.framework.undefined as undefined

class BlobDesc(object):
    def __init__(self, lbi):
        self.lbi_ = lbi
        self.lbn_ = lbi.op_name + "/" + lbi.blob_name
        self.split_axis_ = undefined

    @property
    def lbi(self): return self.lbi_

    @property
    def op_name(self): return self.lbi_.op_name

    @property
    def blob_name(self): return self.lbi_.blob_name

    @property
    def shape(self): return self.static_shape

    @property
    def static_shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def is_dynamic(self):
        raise NotImplementedError

    def split(self, split_axis):
        raise NotImplementedError

    @property
    def split_axis_for_consumer(self):
        assert self.split_axis_ is undefined or type(self.split_axis_) is int \
            or self.split_axis_ is None or self.split_axis_ == False
        return self.split_axis_
    
    def has_split_axis_for_consumer(self):
        return self.split_axis_ is not undefined

    @property
    def logical_blob_name(self):
        if self.split_axis_ is undefined:
            return self.lbn_
        elif type(self.split_axis_) is int:
            return self.lbn_ + ":S" + str(self.split_axis_)
        elif self.split_axis_ is None or self.split_axis_ is False:
            return self.lbn_ + ":B"
        else:
            raise NotImplementedError


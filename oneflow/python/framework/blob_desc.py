from __future__ import absolute_import

import oneflow.core.common.data_type_pb2 as data_type_util

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
    def has_batch_dim(self):
        raise NotImplementedError

    @property
    def is_dynamic(self):
        raise NotImplementedError

    def split(self, split_axis):
        ret = copy.deepcopy(self)
        ret.split_axis_ = split_axis
        return ret

    @property
    def has_split_axis(self):
        raise NotImplementedError

    @property
    def split_axis(self):
        raise NotImplementedError

    @property
    def logical_blob_name(self):
        if self.split_axis_ == undefined:
            return lbn_
        elif type(self.split_axis_ is int):
            return lbi_ + "S" + str(self.split_axis_)
        elif split_axis_ is None or (type(self.split_axis_ is bool) and not self.split_axis):
            return lbi_ + "B"
        else:
            raise NotImplementedError


from __future__ import absolute_import

import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.python.framework.distribute as distribute_util

class BlobDesc(object):
    def __init__(self, lbi):
        self.lbi_ = lbi
        self.lbn_ = lbi.op_name + "/" + lbi.blob_name
        self.distribute_ = distribute_util.auto()

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
    def batch_axis(self):
        raise NotImplementedError

    @property
    def is_dynamic(self):
        raise NotImplementedError

    def with_distribute(self, distribute):
        raise NotImplementedError

    def with_split_distribute(self, axis):
        return self.with_distribute(distribute_util.split(axis))

    def with_broadcast_distribute(self):
        return self.with_distribute(distribute_util.broadcast())

    @property
    def distribute(self):
        distribute_util.assert_is_valid_distribute(self.distribute_)
        return self.distribute_

    @property
    def logical_blob_name(self):
        if type(self.distribute_) is distribute_util.AutoDistribute:
            return self.lbn_
        elif type(self.distribute_) is distribute_util.SplitDistribute:
            return self.lbn_ + ":S" + str(self.distribute_.axis)
        elif type(self.distribute_) is distribute_util.BroadcastDistribute:
            return self.lbn_ + ":B"
        else:
            raise NotImplementedError


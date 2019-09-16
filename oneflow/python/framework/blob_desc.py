from __future__ import absolute_import

import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.python.framework.parallel as parallel_util

class BlobDesc(object):
    def __init__(self, lbi):
        self.lbi_ = lbi
        self.lbn_ = lbi.op_name + "/" + lbi.blob_name
        self.parallel_for_consumer_ = parallel_util.auto()

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

    def parallel(self, parallel):
        raise NotImplementedError

    @property
    def parallel_for_consumer(self):
        parallel_util.assert_is_valid_parallel(self.parallel_for_consumer_)
        return self.parallel_for_consumer_
    
    @property
    def logical_blob_name(self):
        if type(self.parallel_for_consumer_) is parallel_util.AutoParallel:
            return self.lbn_
        elif type(self.parallel_for_consumer_) is parallel_util.SplitParallel:
            return self.lbn_ + ":S" + str(self.parallel_for_consumer_.axis)
        elif type(self.parallel_for_consumer_) is parallel_util.BroadcastParallel:
            return self.lbn_ + ":B"
        else:
            raise NotImplementedError


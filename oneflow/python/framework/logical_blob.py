from __future__ import absolute_import

import oneflow.python.framework.inter_user_job as inter_user_job

def LogicalBlob:
    def __init__(self, shape, dtype,
                 has_batch_dim = True,
                 is_dynamic = False,
                 split_axis = None,
                 broadcast = None):
        self.shape_ = shape
        self.dtype_ = dtype
        self.has_batch_dim_ = has_batch_dim
        self.is_dynamic_ = is_dynamic
        self.split_axis_ = split_axis
        self.broadcast_ = broadcast
    
    @property
    def op_name(self):
        return self.op_name_

    def set_op_name(self, op_name):
        self.op_name_ = op_name

    @property
    def shape(self):
        return self.shape_

    @property
    def dtype(self):
        return self.dtype_

    @property
    def has_batch_dim(self):
        return self.has_batch_dim_

    @property
    def is_dynamic(self):
        return self.is_dynamic_

    @property
    def split_axis(self):
        return self.split_axis_

    @property
    def broadcast(self):
        return self.broadcast_

    def pull(self):
        return inter_user_job.pull(self)

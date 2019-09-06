from __future__ import absolute_import

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.python.framework.inter_user_job_util as inter_user_job_util
import oneflow.python.framework.job_builder as job_builder
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow

class RemoteBlob(blob_desc.BlobDesc):
    def __init__(self, lbi):
        self.job_name_ = job_builder.GetCurCtxJobName()
        self.lbi_ = lbi
        self.lbn_ = lbi.op_name + "/" + lbi.blob_name
        
    @property
    def static_shape(self): return job_builder.GetStaticShape(self.job_name_, self.lbn_)

    @property
    def dtype(self): return job_builder.GetDataType(self.job_name_, self.lbn_)

    @property
    def has_batch_dim(self): return job_builder.GetHasBatchDim(self.job_name_, self.lbn_)
    
    @property
    def op_name(self):
        return self.lbi_.op_name

    @property
    def logical_blob_name(self):
        return "%s/%s" % (self.lbi_.op_name, self.lbi_.blob_name)

    def pull(self):
        return inter_user_job_util.pull(self)

    def __add__(self, rhs): 
        return oneflow.math.add(self, rhs)
    
    def __radd__(self, lhs): 
        return oneflow.math.add(lhs, self)

    def __sub__(self, rhs):
        return oneflow.math.subtract(self, rhs)

    def __rsub__(self, lhs):
        return oneflow.math.subtract(lhs, self)

    def __mul__(self, rhs):
        return oneflow.math.multiply(self, rhs)

    def __rmul__(self, lhs):
        return oneflow.math.multiply(lhs, self)

    def __mul__(self, rhs):
        return oneflow.math.multiply(self, rhs)

    def __rmul__(self, lhs):
        return oneflow.math.multiply(lhs, self)

    def __truediv__(self, rhs):
        return oneflow.math.divide(self, rhs)

    def __div__(self, rhs):
        return oneflow.math.divide(self, rhs)

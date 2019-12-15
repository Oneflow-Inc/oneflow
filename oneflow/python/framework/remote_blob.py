from __future__ import absolute_import

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.python.framework.watch_scope_util as watch_scope_util
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.python.framework.c_api_util as c_api_util

import oneflow

class RemoteBlob(blob_desc.BlobDesc):
    def __init__(self, lbi, **kw):
        blob_desc.BlobDesc.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        watch_scope_util.TryWatchOnce(self)

    @property
    def static_shape(self):
        return c_api_util.JobBuildAndInferCtx_GetStaticShape(self.job_name_, self.lbn_)

    @property
    def dtype(self):
        return c_api_util.JobBuildAndInferCtx_GetDataType(self.job_name_, self.lbn_)

    @property
    def batch_axis(self):
        return c_api_util.JobBuildAndInferCtx_GetBatchAxis(self.job_name_, self.lbn_)

    @property
    def disable_boxing(self):
        return c_api_util.JobBuildAndInferCtx_DisableBoxing(self.job_name_, self.lbn_)
    
    @property
    def parallel_conf(self):
        return c_api_util.JobBuildAndInferCtx_GetParallelConfFromProducerView(self.job_name_,
                                                                              self.lbn_)

    def with_distribute(self, distribute):
        oneflow.distribute.assert_is_valid_distribute(distribute)
        ret = RemoteBlob(self.lbi_)
        ret.distribute_ = distribute
        return ret

    def with_gradient_distribute(self, distribute):
        return oneflow.parallel_cast(self, gradient_distribute=distribute)

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

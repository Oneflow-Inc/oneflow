from __future__ import absolute_import

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.python.framework.watch_scope_util as watch_scope_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.python.framework.c_api_util as c_api_util

import oneflow

def RemoteBlob(lbi, **kw):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    lbn = lbi.op_name + "/" + lbi.blob_name
    blob_type = ConsistentBlob
    if (c_api_util.JobBuildAndInferCtx_IsMirrorBlob(job_name, lbn)):
        blob_type = MirrorBlob
    return blob_type(lbi, **kw)

class BlobDef(blob_desc.BlobDesc):
    def __init__(self, lbi, **kw):
        blob_desc.BlobDesc.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        self.parallel_size_ = placement_ctx.GetParallelSize(
            placement_ctx.MakeMachineId2DeviceIdList(self.parallel_conf))
        watch_scope_util.TryWatchOnce(self)

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
    
    @property
    def disable_boxing(self):
        raise NotImplementedError
    
    @property
    def is_tensor_list(self):
        raise NotImplementedError

    @property
    def disable_boxing(self):
        raise NotImplementedError
    
    @property
    def parallel_conf(self):
        raise NotImplementedError

    @property
    def parallel_size(self): return self.parallel_size_

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

    def __eq__(self, rhs):
        return oneflow.math.equal(self, rhs)

    def __ne__(self, rhs):
        return oneflow.math.not_equal(self, rhs)

    def __lt__(self, rhs):
        return oneflow.math.less(self, rhs)

    def __le__(self, rhs):
        return oneflow.math.less_equal(self, rhs)

    def __gt__(self, rhs):
        return oneflow.math.greater(self, rhs)

    def __ge__(self, rhs):
        return oneflow.math.greater_equal(self, rhs)

class ConsistentBlob(BlobDef):
    def __init__(self, lbi, **kw):
        BlobDef.__init__(self, lbi, **kw)
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
    def is_dynamic(self):
        return c_api_util.JobBuildAndInferCtx_IsDynamic(self.job_name_, self.lbn_)
    
    @property
    def disable_boxing(self):
        return c_api_util.JobBuildAndInferCtx_DisableBoxing(self.job_name_, self.lbn_)
    
    @property
    def is_tensor_list(self):
        return c_api_util.JobBuildAndInferCtx_IsTensorList(self.job_name_, self.lbn_)

    @property
    def disable_boxing(self):
        return c_api_util.JobBuildAndInferCtx_DisableBoxing(self.job_name_, self.lbn_)
    
    @property
    def parallel_conf(self):
        return c_api_util.JobBuildAndInferCtx_GetParallelConfFromProducerView(self.job_name_,
                                                                              self.lbn_)

class MirrorBlob(BlobDef):
    def __init__(self, lbi, **kw):
        BlobDef.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        watch_scope_util.TryWatchOnce(self)
        self.sub_consistent_blob_list_ = []
        lbn = self.logical_blob_name
        num_sub_lbi = c_api_util.JobBuildAndInferCtx_MirrorBlobGetNumSubLbi(self.job_name_, lbn)
        for i in range(num_sub_lbi):
            sub_lbi = c_api_util.JobBuildAndInferCtx_MirrorBlobGetSubLbi(self.job_name_, lbn, i)
            self.sub_consistent_blob_list_.append(ConsistentBlob(sub_lbi))
            
    @property       
    def sub_consistent_blob_list(self): return self.sub_consistent_blob_list_
        
    @property
    def static_shape(self):
        return c_api_util.JobBuildAndInferCtx_MirrorBlobGetStaticShape(self.job_name_, self.lbn_)

    @property
    def dtype(self):
        return c_api_util.JobBuildAndInferCtx_MirrorBlobGetDataType(self.job_name_, self.lbn_)

    @property
    def batch_axis(self):
        return c_api_util.JobBuildAndInferCtx_MirrorBlobGetBatchAxis(self.job_name_, self.lbn_)

    @property
    def is_dynamic(self):
        return c_api_util.JobBuildAndInferCtx_MirrorBlobIsDynamic(self.job_name_, self.lbn_)
    
    @property
    def disable_boxing(self):
        return c_api_util.JobBuildAndInferCtx_MirrorBlobDisableBoxing(self.job_name_, self.lbn_)
    
    @property
    def is_tensor_list(self):
        return c_api_util.JobBuildAndInferCtx_MirrorBlobIsTensorList(self.job_name_, self.lbn_)

    @property
    def disable_boxing(self):
        return c_api_util.JobBuildAndInferCtx_MirrorBlobDisableBoxing(self.job_name_, self.lbn_)
    
    @property
    def parallel_conf(self):
        return c_api_util.JobBuildAndInferCtx_MirrorBlobGetParallelConfFromProducerView(
                self.job_name_, self.lbn_)

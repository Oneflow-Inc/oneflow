from __future__ import absolute_import

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.python.framework.watch_scope_util as watch_scope_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.blob_trait as blob_trait
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.hob as hob

import oneflow

def RemoteBlob(lbi, **kw):
    return enable_if.unique(EagerRemoteBlob, LazyRemoteBlob)(lbi, **kw)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def EagerRemoteBlob(lbi, **kw):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    lbn = lbi.op_name + "/" + lbi.blob_name
    if (c_api_util.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn)):
        return EagerMirroredBlob(lbi, **kw)
    else:
        raise NotImplementedError("EagerConsistentBlob not supported yet")

@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def LazyRemoteBlob(lbi, **kw):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    lbn = lbi.op_name + "/" + lbi.blob_name
    blob_type = LazyConsistentBlob
    if (c_api_util.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn)):
        blob_type = LazyMirroredBlob
    return blob_type(lbi, **kw)


class BlobDef(blob_desc.BlobDesc, blob_trait.BlobOperatorTrait, blob_trait.BlobHeaderTrait):
    def __init__(self, lbi, **kw):
        blob_desc.BlobDesc.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()

    @property
    def batch_axis(self):
        raise NotImplementedError

    @property
    def split_axis(self):
        raise NotImplementedError

    @property
    def disable_boxing(self):
        raise NotImplementedError

    @property
    def parallel_conf(self):
        raise NotImplementedError

    @property
    def parallel_size(self):
        return placement_ctx.GetParallelSize(
            placement_ctx.MakeMachineId2DeviceIdList(self.parallel_conf))

    def with_distribute(self, distribute):
        oneflow.distribute.assert_is_valid_distribute(distribute)
        ret = RemoteBlob(self.lbi_)
        ret.distribute_ = distribute
        return ret

    def with_gradient_distribute(self, distribute):
        return oneflow.parallel_cast(self, gradient_distribute=distribute)

class ConsistentBlob(BlobDef):
    def __init__(self, *args, **kwargs):
        BlobDef.__init__(self, *args, **kwargs)

class LazyConsistentBlob(ConsistentBlob):
    def __init__(self, lbi, auto_watched_within_scope = True, **kw):
        ConsistentBlob.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        if auto_watched_within_scope: watch_scope_util.TryWatchOnce(self)

    @property
    def shape(self):
        return c_api_util.JobBuildAndInferCtx_GetStaticShape(self.job_name_, self.lbn_)

    @property
    def dtype(self):
        return c_api_util.JobBuildAndInferCtx_GetDataType(self.job_name_, self.lbn_)

    @property
    def batch_axis(self):
        return c_api_util.JobBuildAndInferCtx_GetBatchAxis(self.job_name_, self.lbn_)

    @property
    def split_axis(self):
        return c_api_util.JobBuildAndInferCtx_GetSplitAxisFromProducerView(
                            self.job_name_, self.lbn_)

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
    def parallel_conf(self):
        return c_api_util.JobBuildAndInferCtx_GetParallelConfFromProducerView(self.job_name_,
                                                                              self.lbn_)

class MirroredBlob(BlobDef):
    def __init__(self, *args, **kwargs):
        BlobDef.__init__(self, *args, **kwargs)

class LazyMirroredBlob(MirroredBlob):
    def __init__(self, lbi, **kw):
        MirroredBlob.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        self.sub_consistent_blob_list_ = []
        lbn = self.unique_name
        num_sub_lbi = c_api_util.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(self.job_name_, lbn)
        for i in range(num_sub_lbi):
            sub_lbi = c_api_util.JobBuildAndInferCtx_MirroredBlobGetSubLbi(self.job_name_, lbn, i)
            consistent_blob = LazyConsistentBlob(sub_lbi, auto_watched_within_scope=False)
            self.sub_consistent_blob_list_.append(consistent_blob)
        watch_scope_util.TryWatchOnce(self)

    @property
    def sub_consistent_blob_list(self): return self.sub_consistent_blob_list_

    @property
    def shape(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetStaticShape(self.job_name_, self.lbn_)

    @property
    def dtype(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetDataType(self.job_name_, self.lbn_)

    @property
    def batch_axis(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetBatchAxis(self.job_name_, self.lbn_)

    @property
    def split_axis(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(
                            self.job_name_, self.lbn_)

    @property
    def is_dynamic(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobIsDynamic(self.job_name_, self.lbn_)

    @property
    def disable_boxing(self): return True

    @property
    def is_tensor_list(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobIsTensorList(self.job_name_, self.lbn_)

    @property
    def parallel_conf(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetParallelConfFromProducerView(
                self.job_name_, self.lbn_)

class EagerMirroredBlob(MirroredBlob):
    def __init__(self, lbi, **kw):
        MirroredBlob.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        self.sub_consistent_blob_list_ = []
        self.shape_ = c_api_util.JobBuildAndInferCtx_MirroredBlobGetStaticShape(
                self.job_name_, self.lbn_)
        self.dtype_ = c_api_util.JobBuildAndInferCtx_MirroredBlobGetDataType(
                self.job_name_, self.lbn_)
        self.batch_axis_ = c_api_util.JobBuildAndInferCtx_MirroredBlobGetBatchAxis(
                self.job_name_, self.lbn_)
        self.split_axis_ = c_api_util.JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(
                self.job_name_, self.lbn_)
        self.is_tensor_list_ = c_api_util.JobBuildAndInferCtx_MirroredBlobIsTensorList(
                self.job_name_, self.lbn_)
        self.parallel_conf_ = (
                c_api_util.JobBuildAndInferCtx_MirroredBlobGetParallelConfFromProducerView(
                    self.job_name_, self.lbn_))

    @property
    def sub_consistent_blob_list(self): raise NotImplementedError

    @property
    def shape(self): return self.shape_

    @property
    def dtype(self): return self.dtype_

    @property
    def batch_axis(self): return self.batch_axis_

    @property
    def split_axis(self): return self.split_axis_

    @property
    def is_dynamic(self): return True

    @property
    def disable_boxing(self): return True

    @property
    def is_tensor_list(self): return self.is_tensor_list_

    @property
    def parallel_conf(self): return self.parallel_conf_


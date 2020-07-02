from __future__ import absolute_import

import oneflow
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.python.framework.blob_desc as blob_desc
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.watch_scope_util as watch_scope_util
import oneflow.python.framework.blob_trait as blob_trait
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.hob as hob
import oneflow.python.eager.eager_blob_util as eager_blob_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.gradient_util as gradient_util

blob_register = blob_register_util.GetDefaultBlobRegister()


def RemoteBlob(lbi, **kw):
    api = enable_if.unique([EagerLogicalBlob, LazyRemoteBlob])
    return api(lbi, **kw)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def EagerLogicalBlob(lbi, **kw):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    lbn = lbi.op_name + "/" + lbi.blob_name
    if c_api_util.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn):
        return EagerMirroredBlob(lbi, **kw)
    else:
        return EagerConsistentBlob(lbi, **kw)


@enable_if.condition(~hob.eager_execution_enabled)
def LazyRemoteBlob(lbi, **kw):
    job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
    lbn = lbi.op_name + "/" + lbi.blob_name
    blob_type = LazyConsistentBlob
    if c_api_util.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn):
        blob_type = LazyMirroredBlob
    return blob_type(lbi, **kw)


class BlobDef(
    blob_desc.BlobDesc, blob_trait.BlobOperatorTrait, blob_trait.BlobHeaderTrait
):
    def __init__(self, lbi, **kw):
        blob_desc.BlobDesc.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        self.parallel_size_ = 0

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
        if self.parallel_size_ == 0:
            self.parallel_size_ = placement_ctx.GetParallelSize(
                placement_ctx.MakeMachineId2DeviceIdList(self.parallel_conf)
            )
        return self.parallel_size_

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
    def __init__(self, lbi, auto_watched_within_scope=True, **kw):
        ConsistentBlob.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        if auto_watched_within_scope:
            watch_scope_util.TryWatchOnce(self)

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
            self.job_name_, self.lbn_
        )

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
        return c_api_util.JobBuildAndInferCtx_GetParallelConfFromProducerView(
            self.job_name_, self.lbn_
        )


class MirroredBlob(BlobDef):
    def __init__(self, *args, **kwargs):
        BlobDef.__init__(self, *args, **kwargs)


class LazyMirroredBlob(MirroredBlob):
    def __init__(self, lbi, **kw):
        MirroredBlob.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        self.sub_consistent_blob_list_ = []
        lbn = self.unique_name
        num_sub_lbi = c_api_util.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(
            self.job_name_, lbn
        )
        for i in range(num_sub_lbi):
            sub_lbi = c_api_util.JobBuildAndInferCtx_MirroredBlobGetSubLbi(
                self.job_name_, lbn, i
            )
            consistent_blob = LazyConsistentBlob(
                sub_lbi, auto_watched_within_scope=False
            )
            self.sub_consistent_blob_list_.append(consistent_blob)
        watch_scope_util.TryWatchOnce(self)

    def numpy_mirrored_list(self):
        return []

    @property
    def sub_consistent_blob_list(self):
        return self.sub_consistent_blob_list_

    @property
    def shape(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetStaticShape(
            self.job_name_, self.lbn_
        )

    @property
    def dtype(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetDataType(
            self.job_name_, self.lbn_
        )

    @property
    def batch_axis(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetBatchAxis(
            self.job_name_, self.lbn_
        )

    @property
    def split_axis(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(
            self.job_name_, self.lbn_
        )

    @property
    def is_dynamic(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobIsDynamic(
            self.job_name_, self.lbn_
        )

    @property
    def disable_boxing(self):
        return True

    @property
    def is_tensor_list(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobIsTensorList(
            self.job_name_, self.lbn_
        )

    @property
    def parallel_conf(self):
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetParallelConfFromProducerView(
            self.job_name_, self.lbn_
        )


class EagerBlobMixin(object):
    @property
    def blob_object(self):
        return self.blob_object_

    def numpy(self, rank):
        raise NotImplementedError

    def numpy_list(self, rank):
        raise NotImplementedError

    def numpy_mirrored_list(self):
        raise NotImplementedError

    @property
    def sub_consistent_blob_list(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.blob_object_.op_arg_blob_attr.shape

    @property
    def dtype(self):
        return self.blob_object_.op_arg_blob_attr.dtype

    @property
    def batch_axis(self):
        opt_batch_axis = self.blob_object_.op_arg_blob_attr.batch_axis
        if opt_batch_axis.HasField("value"):
            return opt_batch_axis.value
        else:
            return None

    @property
    def split_axis(self):
        sbp_parallel = self.blob_object_.op_arg_parallel_attr.sbp_parallel
        if sbp_parallel.HasField("split_parallel"):
            return sbp_parallel.split_parallel.axis
        elif sbp_parallel.HasField("broadcast_parallel"):
            return None
        else:
            raise NotImplementedError

    @property
    def is_dynamic(self):
        return True

    @property
    def disable_boxing(self):
        return True

    @property
    def is_tensor_list(self):
        return self.blob_object_.op_arg_blob_attr.is_tensor_list

    @property
    def parallel_conf(self):
        return self.blob_object_.parallel_desc_symbol.parallel_conf

    def __del__(self):
        blob_register.ClearObject4BlobName(self.unique_name)

    def _Init(self, blob_object):
        if blob_object is None:
            self.blob_object_ = blob_register.GetObject4BlobName(self.unique_name)
        else:
            blob_register.SetObject4BlobName(self.unique_name, blob_object)
            self.blob_object_ = blob_object
        self.sub_consistent_blob_list_ = []


class EagerConsistentBlob(EagerBlobMixin, ConsistentBlob):
    def __init__(self, lbi, blob_object=None, **kw):
        ConsistentBlob.__init__(self, lbi, **kw)
        self._Init(blob_object)

    def numpy(self, rank):
        assert rank is None
        assert self.is_tensor_list is not True
        raise NotImplementedError


class EagerMirroredBlob(EagerBlobMixin, MirroredBlob):
    def __init__(self, lbi, blob_object=None, **kw):
        MirroredBlob.__init__(self, lbi, **kw)
        self._Init(blob_object)

    def numpy(self, rank):
        assert rank is not None
        assert self.is_tensor_list is not True

        ndarray_list = self.numpy_mirrored_list()
        return ndarray_list[rank]

    def numpy_mirrored_list(self):
        assert self.is_tensor_list is not True
        physical_blob_names = []

        def UnpackLogicalBlobToPhysicalBlobs(builder):
            physical_blob_objects = builder.UnpackLogicalBlobToPhysicalBlobs(
                self.blob_object_
            )
            for i, physical_blob_object in enumerate(physical_blob_objects):
                blob_name = "{}/{}".format(self.unique_name, i)
                physical_blob_names.append(blob_name)
                if not blob_register.HasObject4BlobName(blob_name):
                    blob_register.SetObject4BlobName(blob_name, physical_blob_object)

        def FetchBlobNumpyMirroredList(blob_object):
            vm_util.LogicalRun(UnpackLogicalBlobToPhysicalBlobs)
            return [
                eager_blob_util.EagerPhysicalBlob(name).numpy()
                for name in physical_blob_names
            ]

        blob_cache = blob_cache_util.FindOrCreateBlobCache(self.blob_object_)
        return blob_cache.GetCachedNumpyMirroredList(FetchBlobNumpyMirroredList)

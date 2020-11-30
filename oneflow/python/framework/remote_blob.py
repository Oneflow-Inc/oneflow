"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

import oneflow
import oneflow.python.framework.blob_desc as blob_desc
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.blob_trait as blob_trait
from oneflow.python.framework.dtype import convert_proto_dtype_to_oneflow_dtype
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.hob as hob
import oneflow.python.eager.eager_blob_util as eager_blob_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.gradient_util as gradient_util
import oneflow.python.eager.boxing_util as boxing_util
import oneflow.python.framework.op_arg_util as op_arg_util
import oneflow_api.oneflow.core.job.placement as placement_cfg
import traceback
import sys

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
    def __init__(self, lbi, job_name=None, **kw):
        blob_desc.BlobDesc.__init__(self, lbi, **kw)
        if job_name is None:
            job_name = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        self.job_name_ = job_name
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
    def __init__(self, lbi, **kw):
        ConsistentBlob.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()

    @property
    def shape(self):
        if oneflow.scope.mirrored_view_enabled():
            print(
                "WARNING:",
                "You access a consistent blob shape in mirrored view, there may be problems,",
                "you should add 'x = flow.cast_to_current_logical_view(x)'.",
                file=sys.stderr,
            )
            print(traceback.format_stack()[-2])
        return c_api_util.JobBuildAndInferCtx_GetStaticShape(self.job_name_, self.lbn_)

    @property
    def dtype(self):
        return convert_proto_dtype_to_oneflow_dtype(
            c_api_util.JobBuildAndInferCtx_GetDataType(self.job_name_, self.lbn_)
        )

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

    def IdenticalTo(self, rhs):
        return (
            self.unique_name == rhs.unique_name
            and self.shape == rhs.shape
            and self.batch_axis == rhs.batch_axis
            and self.split_axis == rhs.split_axis
            and self.is_dynamic == rhs.is_dynamic
            and self.disable_boxing == rhs.disable_boxing
            and self.is_tensor_list == rhs.is_tensor_list
            and self.parallel_conf == rhs.parallel_conf
        )


class MirroredBlob(BlobDef):
    def __init__(self, *args, **kwargs):
        BlobDef.__init__(self, *args, **kwargs)


class LazyMirroredBlob(MirroredBlob):
    def __init__(self, lbi, **kw):
        MirroredBlob.__init__(self, lbi, **kw)
        self.job_name_ = c_api_util.JobBuildAndInferCtx_GetCurrentJobName()
        self.sub_consistent_blob_list_ = []
        lbn = self.logical_blob_name
        num_sub_lbi = c_api_util.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(
            self.job_name_, lbn
        )
        for i in range(num_sub_lbi):
            sub_lbi = c_api_util.JobBuildAndInferCtx_MirroredBlobGetSubLbi(
                self.job_name_, lbn, i
            )
            consistent_blob = LazyConsistentBlob(sub_lbi)
            self.sub_consistent_blob_list_.append(consistent_blob)

    @property
    def sub_consistent_blob_list(self):
        return self.sub_consistent_blob_list_

    @property
    def shape(self):
        if oneflow.scope.consistent_view_enabled():
            print(
                "WARNING:",
                "You access a mirrored blob shape in consistent view, there may be problems,"
                "you should add 'x = flow.cast_to_current_logical_view(x)'.",
                file=sys.stderr,
            )
            print(traceback.format_stack()[-2])
        return c_api_util.JobBuildAndInferCtx_MirroredBlobGetStaticShape(
            self.job_name_, self.lbn_
        )

    @property
    def dtype(self):
        return convert_proto_dtype_to_oneflow_dtype(
            c_api_util.JobBuildAndInferCtx_MirroredBlobGetDataType(
                self.job_name_, self.lbn_
            )
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


class EagerBlobTrait(object):
    def numpy_size(self):
        return self.blob_object.parallel_desc_symbol.parallel_num

    def numpy_list_size(self):
        return self.blob_object.parallel_desc_symbol.parallel_num

    def numpy(self, rank=None):
        if rank is None:
            if self.numpy_size() == 1:
                return self._NumpyAt(0)
            else:
                assert not self.is_dynamic
                assert not self.is_tensor_list
                return self._Numpy()
        else:
            return self._NumpyAt(rank)

    def numpy_list(self, rank=None):
        assert self.is_tensor_list
        assert self.is_dynamic
        mirrored_list = self._NumpyMirroredList()
        if rank is None:
            return mirrored_list
        else:
            parallel_num = self.blob_object_.parallel_desc_symbol.parallel_num
            assert rank >= 0
            assert rank < parallel_num
            assert len(mirrored_list) == parallel_num
            return mirrored_list[rank]

    @property
    def sub_consistent_blob_list(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.blob_object.op_arg_blob_attr.shape

    @property
    def dtype(self):
        ret = self.blob_object.op_arg_blob_attr.dtype
        assert issubclass(ret, dtype_util.dtype)
        return ret

    @property
    def batch_axis(self):
        opt_batch_axis = self.blob_object.op_arg_blob_attr.batch_axis
        if opt_batch_axis.HasField("value"):
            return opt_batch_axis.value
        else:
            return None

    @property
    def split_axis(self):
        sbp_parallel = self.blob_object.op_arg_parallel_attr.sbp_parallel
        if sbp_parallel.HasField("split_parallel"):
            return sbp_parallel.split_parallel.axis
        elif sbp_parallel.HasField("broadcast_parallel"):
            return None
        elif sbp_parallel.HasField("partial_sum_parallel"):
            return None
        else:
            raise NotImplementedError

    @property
    def is_dynamic(self):
        return self.blob_object.op_arg_blob_attr.is_dynamic

    @property
    def disable_boxing(self):
        return True

    @property
    def is_tensor_list(self):
        return self.blob_object.op_arg_blob_attr.is_tensor_list

    @property
    def parallel_conf(self):
        return self.blob_object.parallel_desc_symbol.parallel_conf

    def __del__(self):
        blob_register.CloseRegisteredBlobAccess(self.logical_blob_name)

    def _Init(self, blob_object):
        access = blob_register.OpenRegisteredBlobAccess(
            self.logical_blob_name, blob_object
        )
        self.registered_blob_access_ = access
        self.sub_consistent_blob_list_ = []

    @property
    def blob_object(self):
        return self.registered_blob_access_.blob_object

    def _NumpyAt(self, rank):
        assert self.is_tensor_list is not True
        assert rank >= 0
        assert rank < self.blob_object.parallel_desc_symbol.parallel_num
        ndarray_list = self._NumpyMirroredList()
        return ndarray_list[rank]

    def _Numpy(self):
        assert self.is_tensor_list is not True

        def FetchBlobNumpy(blob_object):
            consistent_blob_name = None

            def BoxingToSingleDevice(builder):
                parallel_conf = placement_cfg.ParallelConf()
                parallel_conf.set_device_tag(
                    blob_object.parallel_desc_symbol.device_tag
                )
                parallel_conf.add_device_name("{}:{}".format(0, 0))
                tmp_parallel_desc_symbol = builder.GetParallelDescSymbol(parallel_conf)
                tmp_op_arg_parallel_attr = op_arg_util.OpArgParallelAttribute(
                    tmp_parallel_desc_symbol,
                    blob_object.op_arg_parallel_attr.sbp_parallel,
                    blob_object.op_arg_parallel_attr.opt_mirrored_parallel,
                )
                with oneflow.scope.placement(
                    self.parallel_conf.device_tag(),
                    list(self.parallel_conf.device_name()),
                ):
                    tmp_blob_object = boxing_util.BoxingTo(
                        builder, blob_object, tmp_op_arg_parallel_attr
                    )
                nonlocal consistent_blob_name
                consistent_blob_name = "{}-consistent".format(self.logical_blob_name)
                if not blob_register.HasObject4BlobName(consistent_blob_name):
                    blob_register.SetObject4BlobName(
                        consistent_blob_name, tmp_blob_object
                    )

            vm_util.LogicalRun(BoxingToSingleDevice)
            return eager_blob_util.EagerPhysicalBlob(consistent_blob_name).numpy()

        blob_cache = blob_cache_util.FindOrCreateBlobCache(self.blob_object)
        return blob_cache.GetCachedNumpy(FetchBlobNumpy)

    def _NumpyMirroredList(self):
        physical_blob_objects = []

        def UnpackLogicalBlobToPhysicalBlobs(builder):
            nonlocal physical_blob_objects
            physical_blob_objects = builder.UnpackLogicalBlobToPhysicalBlobs(
                self.blob_object
            )

        def GetPhyBlobNumpy(i, phy_blob_object):
            name = "{}/{}".format(self.logical_blob_name, i)
            blob_register.SetObject4BlobName(name, phy_blob_object)
            return (
                eager_blob_util.EagerPhysicalBlob(name).numpy_list()
                if self.is_tensor_list
                else eager_blob_util.EagerPhysicalBlob(name).numpy()
            )

        def FetchBlobNumpyMirroredList(blob_object):
            vm_util.LogicalRun(UnpackLogicalBlobToPhysicalBlobs)
            return [
                GetPhyBlobNumpy(i, phy_blob_object)
                for i, phy_blob_object in enumerate(physical_blob_objects)
            ]

        blob_cache = blob_cache_util.FindOrCreateBlobCache(self.blob_object)
        return blob_cache.GetCachedNumpyMirroredList(FetchBlobNumpyMirroredList)

    def IdenticalTo(self, rhs):
        return (
            self.blob_object.op_arg_blob_attr == rhs.blob_object.op_arg_blob_attr
            and self.blob_object.op_arg_parallel_attr
            == rhs.blob_object.op_arg_parallel_attr
        )


class EagerConsistentBlob(EagerBlobTrait, ConsistentBlob):
    def __init__(self, lbi, blob_object=None, job_name=None, **kw):
        ConsistentBlob.__init__(self, lbi, job_name=job_name, **kw)
        self._Init(blob_object)


class EagerMirroredBlob(EagerBlobTrait, MirroredBlob):
    def __init__(self, lbi, blob_object=None, **kw):
        MirroredBlob.__init__(self, lbi, **kw)
        self._Init(blob_object)

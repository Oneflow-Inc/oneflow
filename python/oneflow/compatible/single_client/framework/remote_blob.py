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
import sys
import traceback

import oneflow._oneflow_internal
from oneflow._oneflow_internal.oneflow.core.job import placement as placement_cfg
from oneflow._oneflow_internal.oneflow.core.register import logical_blob_id as lbi_util
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.eager import boxing_util as boxing_util
from oneflow.compatible.single_client.eager import eager_blob_util as eager_blob_util
from oneflow.compatible.single_client.eager import gradient_util as gradient_util
from oneflow.compatible.single_client.framework import blob_trait as blob_trait
from oneflow.compatible.single_client.framework import c_api_util as c_api_util
from oneflow.compatible.single_client.framework import hob as hob
from oneflow.compatible.single_client.framework import id_util as id_util
from oneflow.compatible.single_client.framework import (
    placement_context as placement_ctx,
)
from oneflow.compatible.single_client.framework.dtype import (
    convert_proto_dtype_to_oneflow_dtype,
)
from oneflow.compatible.single_client.support import enable_if as enable_if
from oneflow.core.register import logical_blob_id_pb2 as logical_blob_id_util

blob_register = oneflow._oneflow_internal.GetDefaultBlobRegister()


def RemoteBlob(lbi, **kw):
    api = enable_if.unique([EagerLogicalBlob, LazyRemoteBlob])
    return api(lbi, **kw)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def EagerLogicalBlob(lbi, **kw):
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    lbn = lbi.op_name + "/" + lbi.blob_name
    if not isinstance(lbi, lbi_util.LogicalBlobId):
        cfg_lbi = lbi_util.LogicalBlobId()
        cfg_lbi.set_op_name(lbi.op_name)
        cfg_lbi.set_blob_name(lbi.blob_name)
        lbi = cfg_lbi
    blob_type = oneflow._oneflow_internal.EagerConsistentBlob
    if c_api_util.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn):
        blob_type = oneflow._oneflow_internal.EagerMirroredBlob
    job_name = ""
    if "job_name" in kw and kw["job_name"] is not None:
        job_name = kw["job_name"]
    blob_object = None
    if "blob_object" in kw:
        blob_object = kw["blob_object"]
    distribute = oneflow._oneflow_internal.distribute.auto()
    if "distribute" in kw:
        distribute = kw["distribute"]
    return blob_type(lbi, blob_object, blob_register, job_name, distribute)


@enable_if.condition(~hob.eager_execution_enabled)
def LazyRemoteBlob(lbi, **kw):
    job_name = oneflow._oneflow_internal.JobBuildAndInferCtx_GetCurrentJobName()
    lbn = lbi.op_name + "/" + lbi.blob_name
    blob_type = oneflow._oneflow_internal.LazyConsistentBlob
    if c_api_util.JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn):
        blob_type = oneflow._oneflow_internal.LazyMirroredBlob
    if not isinstance(lbi, lbi_util.LogicalBlobId):
        cfg_lbi = lbi_util.LogicalBlobId()
        cfg_lbi.set_op_name(lbi.op_name)
        cfg_lbi.set_blob_name(lbi.blob_name)
        lbi = cfg_lbi
    job_name = ""
    if "job_name" in kw and kw["job_name"] is not None:
        job_name = kw["job_name"]
    distribute = oneflow._oneflow_internal.distribute.auto()
    if "distribute" in kw:
        distribute = kw["distribute"]
    return blob_type(lbi, job_name, distribute)


@property
def dtype(self):
    ret = convert_proto_dtype_to_oneflow_dtype(self.get_dtype())
    assert isinstance(ret, flow.dtype)
    return ret


def with_distribute(self, distribute):
    new = type(self)(
        self.lbi, self.job_name, oneflow._oneflow_internal.distribute.auto()
    )
    new.set_distribute(distribute)
    return new


def with_gradient_distribute(self, distribute):
    return flow.parallel_cast(self, gradient_distribute=distribute)


def get_lazy_shape_log_warning(self):
    if flow.scope.mirrored_view_enabled():
        return "%s\n%s\n%s" % (
            "WARNING:",
            "You access a consistent blob shape in mirrored view, there may be problems,",
            "you should add 'x = flow.cast_to_current_logical_view(x)'.",
        )
    else:
        return ""


def get_mirror_shape_log_warning(self):
    if flow.scope.consistent_view_enabled():
        return "%s\n%s\n%s" % (
            "WARNING:",
            "You access a mirrored blob shape in consistent view, there may be problems,",
            "you should add 'x = flow.cast_to_current_logical_view(x)'.",
        )
    else:
        return ""


def RegisterMethod4BlobDef(blob_class):
    blob_class.dtype = dtype
    blob_class.with_distribute = with_distribute
    blob_class.with_gradient_distribute = with_gradient_distribute


def RegisterMethod4LazyConsistentBlob():
    RegisterMethod4BlobDef(oneflow._oneflow_internal.LazyConsistentBlob)
    oneflow._oneflow_internal.LazyConsistentBlob.get_lazy_shape_log_warning = (
        get_lazy_shape_log_warning
    )


def RegisterMethod4LazyMirroredBlob():
    RegisterMethod4BlobDef(oneflow._oneflow_internal.LazyMirroredBlob)
    oneflow._oneflow_internal.LazyMirroredBlob.get_mirror_shape_log_warning = (
        get_mirror_shape_log_warning
    )


@property
def sub_consistent_blob_list(self):
    raise NotImplementedError


def numpy(self, rank=None):
    assert rank is None or rank == 0
    return self._Numpy()


def numpy_list(self, rank=None):
    assert rank is None or rank == 0
    return [self._Numpy()]


def BlobObjectNumpy(blob_object, tmp_name=None):
    if tmp_name is None:
        tmp_name = id_util.UniqueStr("numpy-tmp-")

    def FetchBlobNumpy(blob_object):
        consistent_blob_name = None

        def BoxingToSingleDevice(builder):
            parallel_conf = placement_cfg.ParallelConf()
            parallel_conf.set_device_tag(blob_object.parallel_desc_symbol.device_tag)
            parallel_conf.add_device_name("{}:{}".format(0, 0))
            tmp_parallel_desc_symbol = builder.GetParallelDescSymbol(parallel_conf)
            tmp_op_arg_parallel_attr = oneflow._oneflow_internal.OpArgParallelAttribute(
                tmp_parallel_desc_symbol,
                str(blob_object.op_arg_parallel_attr.sbp_parallel),
                str(blob_object.op_arg_parallel_attr.opt_mirrored_parallel),
            )
            with flow.scope.placement(
                parallel_conf.device_tag(), list(parallel_conf.device_name())
            ):
                tmp_blob_object = boxing_util.BoxingTo(
                    builder, blob_object, tmp_op_arg_parallel_attr
                )
            nonlocal consistent_blob_name
            consistent_blob_name = tmp_name
            if not blob_register.HasObject4BlobName(consistent_blob_name):
                blob_register.SetObject4BlobName(consistent_blob_name, tmp_blob_object)

        oneflow._oneflow_internal.deprecated.LogicalRun(BoxingToSingleDevice)
        return oneflow._oneflow_internal.EagerPhysicalBlob(
            consistent_blob_name,
            blob_register,
            eager_blob_util._GetPhysicalBlobHeaderCache,
        ).numpy()

    return FetchBlobNumpy(blob_object)


def _Numpy(self):
    tmp_name = "{}-consistent".format(self.logical_blob_name)
    return BlobObjectNumpy(self.blob_object, tmp_name)


def RegisterMethod4EagerBlobTrait():
    oneflow._oneflow_internal.EagerBlobTrait.sub_consistent_blob_list = (
        sub_consistent_blob_list
    )
    oneflow._oneflow_internal.EagerBlobTrait.dtype = dtype
    oneflow._oneflow_internal.EagerBlobTrait._Numpy = _Numpy
    oneflow._oneflow_internal.EagerBlobTrait.numpy = numpy
    oneflow._oneflow_internal.EagerBlobTrait.numpy_list = numpy_list


def eager_with_distribute(self, distribute):
    new = type(self)(
        self.lbi,
        blob_object=self.blob_object,
        blob_register=blob_register,
        job_name=self.job_name,
        distribute=self.distribute,
    )
    new.set_distribute(distribute)
    return new


def RegisterMethod4EagerConsistentBlob():
    oneflow._oneflow_internal.EagerConsistentBlob.dtype = dtype
    oneflow._oneflow_internal.EagerConsistentBlob.with_distribute = (
        eager_with_distribute
    )
    oneflow._oneflow_internal.EagerConsistentBlob.with_gradient_distribute = (
        with_gradient_distribute
    )

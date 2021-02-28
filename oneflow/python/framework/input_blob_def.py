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

import sys
from functools import reduce
from typing import Any, Optional, Sequence, Union

import numpy as np

import oneflow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.operator.interface_blob_conf_pb2 as inter_face_blob_conf_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.dtype as dtype_util
from oneflow.python.oneflow_export import oneflow_export
import oneflow_api.oneflow.core.register.logical_blob_id as lbi_util
import oneflow_api
from functools import reduce
import traceback


class ArgBlobDef(object):
    def __init__(
        self, shape, dtype, name=None, distribute=oneflow_api.distribute.auto(),
    ):
        lbi = lbi_util.LogicalBlobId()
        if name is None:
            name = id_util.UniqueStr("Input_")
        lbi.set_op_name(name)
        lbi.set_blob_name("out")
        self.lbi_ = lbi
        assert type(shape) is tuple
        for dim in shape:
            assert type(dim) is int
            assert dim > 0
        self.shape_ = shape
        self.dtype_ = dtype
        self.distribute_ = distribute

    @property
    def lbi(self):
        return self.lbi_

    @property
    def op_name(self):
        return self.lbi_.op_name()

    @property
    def blob_name(self):
        return self.lbi_.blob_name()

    @property
    def unique_name(self):
        return self.op_name + "/" + self.blob_name + self._Distribute2Str()

    @property
    def shape(self):
        return self.shape_

    @property
    def dtype(self):
        return self.dtype_

    @property
    def is_dynamic(self):
        raise NotImplementedError

    @property
    def is_tensor_list(self):
        raise NotImplementedError

    def with_distribute(self, distribute):
        return type(self)(shape=self.shape_, dtype=self.dtype_, name=self.op_name,)

    def Clone(self, op_name=None):
        return type(self)(shape=self.shape_, dtype=self.dtype_, name=op_name,)

    def AddAndInferOp(self, op_conf):
        raise NotImplementedError

    def EagerAddAndInferOp(self, op_conf):
        raise NotImplementedError

    def CheckAndAsyncPush(self, session, arg_ndarray):
        self._CheckNdarray(arg_ndarray)
        self._AsyncPush(session, arg_ndarray)

    def _CheckNdarray(self, ndarray):
        raise NotImplementedError

    def _AsyncPush(self, session, arg_ndarray):
        raise NotImplementedError

    def ToInterfaceBlobConf(self):
        interface_blob_conf = inter_face_blob_conf_util.InterfaceBlobConf()
        interface_blob_conf.shape.dim.extend(self.shape_)
        interface_blob_conf.data_type = self.dtype_.oneflow_proto_dtype
        interface_blob_conf.is_dynamic = self.is_dynamic
        interface_blob_conf.is_tensor_list = self.is_tensor_list
        # NOTE(chengcheng): rm batch_axis, so set split_axis always = 0 for safe. will support
        #     set sbp in future, or will delete in multi-client
        interface_blob_conf.split_axis.value = 0
        return interface_blob_conf

    def _Distribute2Str(self):
        if type(self.distribute_) is oneflow_api.distribute.AutoDistribute:
            return ""
        elif type(self.distribute_) is oneflow_api.distribute.SplitDistribute:
            return ":S" + str(self.distribute_.axis)
        elif type(self.distribute_) is oneflow_api.distribute.BroadcastDistribute:
            return ":B"
        else:
            raise NotImplementedError


class FixedTensorDef(ArgBlobDef):
    def __init__(
        self,
        shape: Sequence[int],
        dtype: dtype_util.dtype = dtype_util.float,
        name: Optional[str] = None,
    ) -> None:
        ArgBlobDef.__init__(
            self, shape, dtype=dtype, name=name,
        )

    @property
    def is_dynamic(self) -> bool:
        return False

    @property
    def is_tensor_list(self) -> bool:
        return False

    def AddAndInferOp(self, op_conf: op_conf_util.OperatorConf) -> Any:
        return compile_context.CurJobAddConsistentOp(op_conf)

    def EagerAddAndInferOp(self, op_conf: op_conf_util.OperatorConf) -> Any:
        parallel_symbol = oneflow.current_scope().device_parallel_desc_symbol
        if (
            parallel_symbol.device_tag == "gpu"
            and list(dict(parallel_symbol.machine_id2device_id_list).keys()) == [0]
            and parallel_symbol.parallel_num == 1
        ):
            device_tag = "gpu"
            device_ids = "0:%s" % (parallel_symbol.machine_id2device_id_list[0][0])
        else:
            device_tag = "cpu"
            device_ids = "0:0"
        with oneflow.scope.placement(device_tag, device_ids):
            return compile_context.CurJobAddConsistentOp(op_conf)

    def _CheckNdarray(self, ndarray: np.ndarray) -> None:
        assert isinstance(ndarray, np.ndarray)
        assert ndarray.shape == self.shape

    def _AsyncPush(self, session: object, arg_ndarray: np.ndarray) -> None:
        session.AsyncPush(self.op_name, _MakePushNdarrayCallback(arg_ndarray))


class MirroredTensorDef(ArgBlobDef):
    def __init__(
        self,
        shape: Sequence[int],
        dtype: dtype_util.dtype = dtype_util.float,
        name: Optional[str] = None,
    ) -> None:
        assert type(shape) is tuple
        ArgBlobDef.__init__(self, shape, dtype=dtype, name=name)
        self.sub_consistent_blob_list_ = []

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def is_tensor_list(self) -> bool:
        return False

    def AddAndInferOp(self, op_conf: op_conf_util.OperatorConf) -> None:
        _AddAndInferMirroredOp(
            self.unique_name, op_conf, self.sub_consistent_blob_list_
        )

    def EagerAddAndInferOp(self, op_conf: op_conf_util.OperatorConf) -> Any:
        return compile_context.CurJobAddMirroredOp(op_conf)

    def _CheckNdarray(self, ndarray_list: Sequence[np.ndarray]) -> None:
        assert isinstance(ndarray_list, (list, tuple))
        assert len(self.sub_consistent_blob_list_) == len(ndarray_list)

        def GetElemCnt(shape):
            return reduce(lambda x, y: x * y, shape, 1)

        for consistent_blob, ndarray in zip(
            self.sub_consistent_blob_list_, ndarray_list
        ):
            assert type(ndarray) is np.ndarray
            assert len(ndarray.shape) == len(self.shape)
            assert GetElemCnt(ndarray.shape) <= GetElemCnt(self.shape)

    def _AsyncPush(self, session: object, ndarray_list: Sequence[np.ndarray]) -> None:
        for i in range(len(ndarray_list)):
            sub_blob = self.sub_consistent_blob_list_[i]
            session.AsyncPush(
                sub_blob.op_name, _MakePushNdarrayCallback(ndarray_list[i])
            )


class MirroredTensorListDef(ArgBlobDef):
    def __init__(
        self,
        shape: Sequence[int],
        dtype: dtype_util.dtype = dtype_util.float,
        name: Optional[str] = None,
    ) -> None:
        assert type(shape) is tuple
        ArgBlobDef.__init__(self, shape, dtype=dtype, name=name)
        self.sub_consistent_blob_list_ = []

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def is_tensor_list(self) -> bool:
        return True

    def AddAndInferOp(self, op_conf: op_conf_util.OperatorConf) -> None:
        _AddAndInferMirroredOp(
            self.unique_name, op_conf, self.sub_consistent_blob_list_
        )

    def EagerAddAndInferOp(self, op_conf: op_conf_util.OperatorConf) -> Any:
        return compile_context.CurJobAddMirroredOp(op_conf)

    def _CheckNdarray(self, ndarray_lists: Sequence[np.ndarray]) -> None:
        assert isinstance(ndarray_lists, (list, tuple))
        assert len(self.sub_consistent_blob_list_) == len(ndarray_lists)

        def GetElemCnt(shape):
            return reduce(lambda x, y: x * y, shape, 1)

        for consistent_blob, ndarray_list in zip(
            self.sub_consistent_blob_list_, ndarray_lists
        ):
            assert type(ndarray_list) is list
            elem_cnt = 0
            for ndarray in ndarray_list:
                assert type(ndarray) is np.ndarray
                assert len(ndarray.shape) == len(self.shape)
                elem_cnt += GetElemCnt(ndarray.shape)
            assert elem_cnt <= GetElemCnt(self.shape)

    def _AsyncPush(self, session: object, ndarray_lists: Sequence[np.ndarray]) -> None:
        for i in range(len(ndarray_lists)):
            sub_blob = self.sub_consistent_blob_list_[i]
            session.AsyncPush(
                sub_blob.op_name, _MakePushNdarrayListCallback(ndarray_lists[i])
            )


def _AddAndInferMirroredOp(mirrored_lbn, op_conf, sub_consistent_blob_list):
    compile_context.CurJobAddMirroredOp(op_conf)
    job_name = oneflow_api.JobBuildAndInferCtx_GetCurrentJobName()
    num_sub_lbi = c_api_util.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(
        job_name, mirrored_lbn
    )
    for i in range(num_sub_lbi):
        sub_lbi = c_api_util.JobBuildAndInferCtx_MirroredBlobGetSubLbi(
            job_name, mirrored_lbn, i
        )
        lbi = lbi_util.LogicalBlobId()
        lbi.set_op_name(sub_lbi.op_name)
        lbi.set_blob_name(sub_lbi.blob_name)
        sub_consistent_blob_list.append(
            oneflow_api.ConsistentBlob(lbi, "", oneflow_api.distribute.auto())
        )


def _MakePushNdarrayCallback(ndarray):
    copied = np.copy(ndarray)

    def Copy(ofblob):
        capacity = reduce(lambda x, y: x * y, ofblob.static_shape, 1)
        elem_cnt = reduce(lambda x, y: x * y, copied.shape, 1)
        assert elem_cnt <= capacity, "%s v.s. %s" % (copied.shape, ofblob.static_shape)
        ofblob.CopyFromNdarray(copied)

    return Copy


def _MakePushNdarrayListCallback(ndarray_list):
    copied = [np.copy(ndarray) for ndarray in ndarray_list]
    return lambda ofblob: ofblob.CopyFromNdarrayList(copied)


@oneflow_export("FixedTensorDef")
class DeprecatedFixedTensorDef(FixedTensorDef):
    def __init__(self, *args, **kwargs):
        running_script = traceback.format_stack()[-2].split(",")[0].split(" ")[3]
        if not running_script.endswith('input_blob_def.py"'):
            print(
                "WARNING: oneflow.FixedTensorDef has been deprecated. "
                "Please use oneflow.typing.Numpy.Placeholder instead."
            )
            print(
                """For instance:
            - def job_func(images=oneflow.FixedTensorDef((32, 1, 28, 28), dtype=flow.float))
            + def job_func(images:oneflow.typing.Numpy.Placeholder((32, 1, 28, 28), dtype=flow.float))"""
            )
            print(traceback.format_stack()[-2])

        super().__init__(*args, **kwargs)


@oneflow_export("MirroredTensorDef")
class DeprecatedMirroredTensorDef(MirroredTensorDef):
    def __init__(self, *args, **kwargs):
        running_script = traceback.format_stack()[-2].split(",")[0].split(" ")[3]
        if not running_script.endswith('input_blob_def.py"'):
            print(
                "WARNING: oneflow.MirroredTensorDef has been deprecated. "
                "Please use oneflow.typing.ListNumpy.Placeholder instead."
            )
            print(
                """For instance:
            - def job_func(images=oneflow.MirroredTensorDef((32, 1, 28, 28), dtype=flow.float))
            + def job_func(images:oneflow.typing.ListNumpy.Placeholder((32, 1, 28, 28), dtype=flow.float))"""
            )
            print(traceback.format_stack()[-2])

        super().__init__(*args, **kwargs)


@oneflow_export("MirroredTensorListDef")
class DeprecatedTensorListDef(MirroredTensorListDef):
    def __init__(self, *args, **kwargs):
        running_script = traceback.format_stack()[-2].split(",")[0].split(" ")[3]
        if not running_script.endswith('input_blob_def.py"'):
            print(
                "WARNING: oneflow.MirroredTensorListDef has been deprecated. "
                "Please use oneflow.typing.ListListNumpy.Placeholder instead."
            )
            print(
                """For instance:
            - def job_func(images=oneflow.MirroredTensorListDef((32, 1, 28, 28), dtype=flow.float))
            + def job_func(images:oneflow.typing.ListListNumpy.Placeholder((32, 1, 28, 28), dtype=flow.float))"""
            )
            print(traceback.format_stack()[-2])

        super().__init__(*args, **kwargs)

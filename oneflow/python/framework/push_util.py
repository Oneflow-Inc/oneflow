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
import oneflow.python.eager.blob_register as blob_register_util
import oneflow.python.framework.input_blob_def as input_blob_def
import oneflow.python.framework.dtype as dtype_util
import oneflow.python.framework.python_callback as python_callback
import oneflow.python.framework.balanced_splitter as balanced_splitter
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.boxing_util as boxing_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow_api.oneflow.core.register.logical_blob_id as lbi_util
import oneflow_api
import numpy
from functools import reduce

blob_register = blob_register_util.GetDefaultBlobRegister()


def AsyncPush(session, job_func, *arg):
    assert len(arg) == len(job_func.__oneflow_input_blob_defs__)
    for i in range(len(arg)):
        _AsyncPushArg(session, job_func.__oneflow_input_blob_defs__[i], arg[i])


def _AsyncPushArg(session, arg_blob_def, arg_ndarray):
    if isinstance(arg_blob_def, (list, tuple)):
        assert isinstance(arg_ndarray, (list, tuple)), "type(arg_ndarray): %s" % (
            type(arg_ndarray)
        )
        assert len(arg_blob_def) == len(arg_ndarray), "%s v.s. %s" % (
            len(arg_blob_def),
            len(arg_ndarray),
        )
        for blob_def, ndarray in zip(arg_blob_def, arg_ndarray):
            _AsyncPushArg(session, blob_def, ndarray)
    elif isinstance(arg_blob_def, dict):
        assert type(arg_blob_def) is type(arg_ndarray)
        assert set(arg_blob_def.keys()) == set(arg_ndarray.keys())
        for k, blob_def in arg_blob_def.items():
            _AsyncPushArg(session, blob_def, arg_ndarray[k])
    else:
        assert isinstance(arg_blob_def, input_blob_def.ArgBlobDef)
        arg_blob_def.CheckAndAsyncPush(session, arg_ndarray)


def MakeEagerInputBlobs(arg_blob_def, arg_ndarray):
    if isinstance(arg_blob_def, (list, tuple)):
        assert isinstance(arg_ndarray, (list, tuple)), "type(arg_ndarray): %s" % (
            type(arg_ndarray)
        )
        assert len(arg_blob_def) == len(arg_ndarray)
        return type(arg_blob_def)(
            MakeEagerInputBlobs(blob_def, ndarray)
            for blob_def, ndarray in zip(arg_blob_def, arg_ndarray)
        )
    elif isinstance(arg_blob_def, dict):
        assert type(arg_blob_def) is type(arg_ndarray)
        assert set(arg_blob_def.keys()) == set(arg_ndarray.keys())
        return {
            k: MakeEagerInputBlobs(blob_def, arg_ndarray[k])
            for k, blob_def in arg_blob_def.items()
        }
    else:
        return _CreateEagerInputBlobAndFeedValue(arg_blob_def, arg_ndarray)


def _CheckInputArgBlobDefValueMatch(arg_blob_def, arg_value):
    if isinstance(arg_blob_def, input_blob_def.FixedTensorDef):
        assert isinstance(arg_value, numpy.ndarray)
        assert arg_blob_def.shape == arg_value.shape
    elif isinstance(arg_blob_def, input_blob_def.MirroredTensorDef):
        assert isinstance(arg_value, (list, tuple))
        for v in arg_value:
            assert isinstance(v, numpy.ndarray)
            assert len(v.shape) == len(arg_blob_def.shape)
            assert numpy.prod(v.shape) <= numpy.prod(arg_blob_def.shape)
    elif isinstance(arg_blob_def, input_blob_def.MirroredTensorListDef):
        assert isinstance(arg_value, (list, tuple))
        for ndarray_list in arg_value:
            for ndarray in ndarray_list:
                assert isinstance(ndarray, numpy.ndarray)
                assert len(ndarray.shape) == len(arg_blob_def.shape)
                assert numpy.prod(ndarray.shape) <= numpy.prod(
                    arg_blob_def.shape
                ), "%s v.s. %s" % (ndarray.shape, arg_blob_def.shape)
    else:
        raise NotImplementedError


def FeedValueToEagerBlob(blob_object, blob_def, ndarray):
    physical_blob_objects = _GetPhysicalBlobObjects(blob_object, None)
    feed_ctx = FeedContext(blob_object.op_arg_parallel_attr, ndarray)
    for i, physical_blob_object in enumerate(physical_blob_objects):
        feed_ctx.set_rank(i)
        _FeedValueToInputPhysicalBlob(feed_ctx, blob_def, physical_blob_object)
    blob_cache_util.TryDisableBlobCache(blob_object)


def _CreateEagerInputBlobAndFeedValue(arg_blob_def, arg_ndarray):
    _CheckInputArgBlobDefValueMatch(arg_blob_def, arg_ndarray)
    arg_blob_object, lbi = _MakeInputBlobObject(arg_blob_def)
    FeedValueToEagerBlob(arg_blob_object, arg_blob_def, arg_ndarray)
    get_blob = None
    if not isinstance(lbi, lbi_util.LogicalBlobId):
        cfg_lbi = lbi_util.LogicalBlobId()
        cfg_lbi.set_op_name(lbi.op_name)
        cfg_lbi.set_blob_name(lbi.blob_name)
        lbi = cfg_lbi
    if isinstance(arg_blob_def, input_blob_def.FixedTensorDef):

        def get_blob(lbi, blob_object, blob_register):
            blob = oneflow_api.EagerConsistentBlob(lbi, blob_object, blob_register)
            with oneflow.scope.consistent_view():
                return oneflow.identity(blob)

    elif isinstance(arg_blob_def, input_blob_def.MirroredTensorDef):
        get_blob = oneflow_api.EagerMirroredBlob
    elif isinstance(arg_blob_def, input_blob_def.MirroredTensorListDef):
        get_blob = oneflow_api.EagerMirroredBlob
    else:
        raise NotImplementedError
    return get_blob(lbi, blob_object=arg_blob_object, blob_register=blob_register)


def _MakeInputBlobObject(arg_blob_def):
    input_op_conf, lbi = _MakeInputOpConfAndRetLbi(arg_blob_def)
    bn_in_op2blob_object = oneflow_api.deprecated.BnInOp2BlobObject()

    def BuildInputInstruction(builder):
        op_attribute = arg_blob_def.EagerAddAndInferOp(input_op_conf)
        scope = oneflow.current_scope()
        parallel_conf = scope.device_parallel_desc_symbol.parallel_conf
        cfg_op_attribute = oneflow_api.deprecated.MakeOpAttributeByString(
            str(op_attribute)
        )
        builder.StatelessCall(
            cfg_op_attribute,
            parallel_conf,
            bn_in_op2blob_object,
            boxing_util.BoxingTo,
            vm_util._FindOrCreateDelegateBlobObject,
        )

    vm_util.LogicalRun(BuildInputInstruction)
    return bn_in_op2blob_object["out"], lbi


def _GetPhysicalBlobObjects(logical_blob_object, lbi):
    blob_register = blob_register_util.GetDefaultBlobRegister()
    physical_blob_objects = None

    def BuildLogical2PhysicalInstruction(builder):
        nonlocal physical_blob_objects
        physical_blob_objects = builder.UnpackLogicalBlobToPhysicalBlobs(
            logical_blob_object
        )

    vm_util.LogicalRun(BuildLogical2PhysicalInstruction)
    return physical_blob_objects


def _MakeInputOpConfAndRetLbi(arg_blob_def):
    assert isinstance(arg_blob_def, input_blob_def.ArgBlobDef)
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("Input_")
    op_conf.input_conf.out = "out"
    op_conf.input_conf.blob_conf.CopyFrom(arg_blob_def.ToInterfaceBlobConf())
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = op_conf.input_conf.out
    return op_conf, lbi


class FeedContext(object):
    def __init__(self, op_arg_parallel_attr, arg_ndarray, rank=0):
        self.op_arg_parallel_attr_ = op_arg_parallel_attr
        self.arg_ndarray_ = arg_ndarray
        self.rank_ = rank
        # balanced_range is used in split_parallel
        self.balanced_range_ = None

    def set_rank(self, rank):
        self.rank_ = rank

    def GetFixedTensor(self, logical_shape):
        assert isinstance(self.arg_ndarray_, numpy.ndarray)
        assert self.arg_ndarray_.shape == logical_shape, "%s v.s. %s" % (
            self.arg_ndarray_.shape,
            logical_shape,
        )
        sbp_parallel = self.op_arg_parallel_attr_.sbp_parallel
        parallel_num = self.op_arg_parallel_attr_.parallel_desc_symbol.parallel_num
        if sbp_parallel.has_broadcast_parallel() or parallel_num == 1:
            return self._AsContiguousNdArray(self.arg_ndarray_)
        elif sbp_parallel.has_split_parallel():
            axis = sbp_parallel.split_parallel().axis()
            start, end = self._GetBalancedRanges(logical_shape[axis])[self.rank_]
            slc = [slice(None)] * len(logical_shape)
            slc[axis] = slice(start, end)
            ndarray = self.arg_ndarray_[tuple(slc)]
            return self._AsContiguousNdArray(ndarray)
        else:
            raise NotImplementedError

    def _GetBalancedRanges(self, dim):
        parallel_num = self.op_arg_parallel_attr_.parallel_desc_symbol.parallel_num
        if self.balanced_range_ is None:
            self.balanced_range_ = balanced_splitter.BalancedRanges(dim, parallel_num)
        return self.balanced_range_

    def GetMirroredTensor(self, static_shape):
        capacity = reduce(lambda x, y: x * y, static_shape, 1)
        assert isinstance(self.arg_ndarray_, (list, tuple))
        parallel_num = self.op_arg_parallel_attr_.parallel_desc_symbol.parallel_num
        assert len(self.arg_ndarray_) == parallel_num
        assert all(isinstance(a, numpy.ndarray) for a in self.arg_ndarray_)
        assert self.rank_ >= 0
        assert self.rank_ < parallel_num
        ndarray = self.arg_ndarray_[self.rank_]
        elem_cnt = reduce(lambda x, y: x * y, ndarray.shape, 1)
        assert elem_cnt <= capacity, "%s v.s. %s" % (ndarray.shape, static_shape)
        return self._AsContiguousNdArray(ndarray)

    def GetMirroredTensorList(self, static_shape):
        assert isinstance(self.arg_ndarray_, (list, tuple))
        parallel_num = self.op_arg_parallel_attr_.parallel_desc_symbol.parallel_num
        assert self.rank_ >= 0
        assert self.rank_ < parallel_num
        assert len(self.arg_ndarray_) == parallel_num
        assert all(isinstance(a, (list, tuple)) for a in self.arg_ndarray_)
        ndarray_list = self.arg_ndarray_[self.rank_]
        assert all(isinstance(arr, numpy.ndarray) for arr in ndarray_list)
        capacity = numpy.prod(static_shape)
        assert all(numpy.prod(arr.shape) <= capacity for arr in ndarray_list)
        return self._AsContiguousNdArray(ndarray_list)

    def _AsContiguousNdArray(self, ndarray):
        if isinstance(ndarray, numpy.ndarray):
            return (
                ndarray if ndarray.data.contiguous else numpy.ascontiguousarray(ndarray)
            )
        elif isinstance(ndarray, (tuple, list)):
            return type(ndarray)(self._AsContiguousNdArray(a) for a in ndarray)
        else:
            raise NotImplementedError


def _FeedValueToInputPhysicalBlob(feed_ctx, blob_def, blob_object):
    assert isinstance(blob_def, input_blob_def.ArgBlobDef)
    assert isinstance(blob_object, oneflow_api.BlobObject)

    FeedBlob = _MakeFeedBlobCallback(feed_ctx, blob_def, blob_object)
    assert callable(FeedBlob)

    def BuildFeedInstruction(builder):
        builder.FeedBlob(
            blob_object, python_callback.GetIdForRegisteredCallback(FeedBlob)
        )
        builder.InsertRemoveForeignCallbackInstruction(
            blob_object.object_id, python_callback.GetIdForRegisteredCallback(FeedBlob)
        )

    vm_util.PhysicalRun(BuildFeedInstruction)


def _MakeFeedBlobCallback(feed_ctx, blob_def, blob_object):
    if isinstance(blob_def, input_blob_def.FixedTensorDef):

        def FeedBlob(ofblob):
            ndarray = feed_ctx.GetFixedTensor(blob_def.shape)
            dtype = dtype_util.convert_oneflow_dtype_to_numpy_dtype(ofblob.dtype)
            assert ndarray.dtype == dtype, "%s v.s. %s" % (ndarray.dtype, dtype)
            assert ndarray.shape == ofblob.static_shape, "%s v.s. %s" % (
                ndarray.shape,
                ofblob.static_shape,
            )
            if ofblob.CopyFromNdarray(ndarray) is False:
                raise ValueError

    elif isinstance(blob_def, input_blob_def.MirroredTensorDef):

        def FeedBlob(ofblob):
            ndarray = feed_ctx.GetMirroredTensor(ofblob.static_shape)
            assert isinstance(ndarray, numpy.ndarray)
            dtype = dtype_util.convert_oneflow_dtype_to_numpy_dtype(ofblob.dtype)
            assert ndarray.dtype == dtype, "%s v.s. %s" % (ndarray.dtype, dtype)
            if ofblob.CopyFromNdarray(ndarray) is False:
                raise ValueError

    elif isinstance(blob_def, input_blob_def.MirroredTensorListDef):

        def FeedBlob(ofblob):
            assert ofblob.is_tensor_list
            ndarray_list = feed_ctx.GetMirroredTensorList(ofblob.static_shape)
            assert isinstance(ndarray_list, (list, tuple))
            assert all(isinstance(ndarray, numpy.ndarray) for ndarray in ndarray_list)
            dtype = dtype_util.convert_oneflow_dtype_to_numpy_dtype(ofblob.dtype)
            assert all(ndarray.dtype == dtype for ndarray in ndarray_list)
            if ofblob.CopyFromNdarrayList(ndarray_list) is False:
                raise ValueError

    else:
        raise NotImplementedError

    return FeedBlob

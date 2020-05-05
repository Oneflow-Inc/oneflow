from __future__ import absolute_import

import sys

import oneflow.python.framework.blob_desc as blob_desc
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as lbi_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.g_func_ctx as g_func_ctx
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.placement_context as placement_ctx
from oneflow.python.oneflow_export import oneflow_export
from functools import reduce
import numpy as np
import oneflow

class ArgBlobDef(blob_desc.BlobDesc):
    def __init__(self, shape, dtype, batch_axis, split_axis='same_with_batch_axis', name=None):
        if split_axis == 'same_with_batch_axis': split_axis = batch_axis
        lbi = lbi_util.LogicalBlobId()
        if name is None: name = id_util.UniqueStr("Input_")
        lbi.op_name = name
        lbi.blob_name = "out"
        blob_desc.BlobDesc.__init__(self, lbi)
        assert type(shape) is tuple
        for dim in shape:
            assert type(dim) is int
            assert dim > 0
        self.shape_ = shape
        self.dtype_ = dtype
        self.batch_axis_ = batch_axis
        self.split_axis_ = split_axis

    @property
    def static_shape(self): return self.shape_

    @property
    def shape(self): return self.shape_

    @property
    def dtype(self): return self.dtype_

    @property
    def batch_axis(self): return self.batch_axis_

    @property
    def is_dynamic(self):
        raise NotImplementedError

    @property
    def is_tensor_list(self):
        raise NotImplementedError

    def with_distribute(self, distribute):
        return type(self)(shape = self.shape_,
                          dtype = self.dtype_,
                          batch_axis = self.batch_axis_,
                          name = self.lbi.op_name)
    
    def Clone(self, op_name = None):
        return type(self)(shape = self.shape_,
                          dtype = self.dtype_,
                          batch_axis = self.batch_axis_,
                          name = op_name)

    def AddAndInferOp(self, op_conf):
        raise NotImplementedError

    def CheckAndAsyncPush(self, session, arg_ndarray):
        self._CheckNdarray(arg_ndarray)
        self._AsyncPush(session, arg_ndarray)
        
    def _CheckNdarray(self, ndarray):
        raise NotImplementedError

    def _AsyncPush(self, session, arg_ndarray):
        raise NotImplementedError

    def ToInterfaceBlobConf(self):
        interface_blob_conf = op_conf_util.InterfaceBlobConf()
        interface_blob_conf.shape.dim.extend(self.shape_)
        interface_blob_conf.data_type = self.dtype_
        interface_blob_conf.is_dynamic = self.is_dynamic
        interface_blob_conf.is_tensor_list = self.is_tensor_list
        if type(self.batch_axis_) is int:
            assert self.batch_axis_ >= 0
            interface_blob_conf.batch_axis.value = self.batch_axis_
        else:
            assert self.batch_axis_ is None or self.batch_axis_ is False
            interface_blob_conf.batch_axis.ClearField("value")
        if type(self.split_axis_) is int:
            assert self.split_axis_ >= 0
            interface_blob_conf.split_axis.value = self.split_axis_
        else:
            assert self.split_axis_ is None or self.split_axis_ is False
            interface_blob_conf.split_axis.ClearField("value")
        return interface_blob_conf

@oneflow_export('FixedTensorDef')
class FixedTensorDef(ArgBlobDef):
    def __init__(self, shape, dtype=data_type_util.kFloat, batch_axis=0,
                 split_axis='same_with_batch_axis', name=None):
        if type(batch_axis) is int:
            if batch_axis < 0: batch_axis += len(shape)
            assert batch_axis >= 0
            assert batch_axis < len(shape)
        ArgBlobDef.__init__(self, shape, dtype=dtype, batch_axis=batch_axis,
                            split_axis=split_axis, name=name)
        
    @property
    def is_dynamic(self): return False

    @property
    def is_tensor_list(self): return False

    def AddAndInferOp(self, op_conf):
        return compile_context.CurJobAddConsistentOp(op_conf)

    def _CheckNdarray(self, ndarray):
        assert isinstance(ndarray, np.ndarray)
        assert ndarray.shape == self.shape

    def _AsyncPush(self, session, arg_ndarray):
        session.AsyncPush(self.op_name, _MakePushNdarrayCallback(arg_ndarray))
        
@oneflow_export('MirroredTensorDef')
class MirroredTensorDef(ArgBlobDef):
    def __init__(self, shape, dtype=data_type_util.kFloat, batch_axis=0, name=None):
        assert type(shape) is tuple
        assert type(batch_axis) is int
        if batch_axis < 0: batch_axis += len(shape)
        assert batch_axis >= 0
        assert batch_axis < len(shape)
        ArgBlobDef.__init__(self, shape, dtype=dtype, batch_axis=batch_axis, name=name)
        self.sub_consistent_blob_list_ = []
        
    @property
    def is_dynamic(self): return True

    @property
    def is_tensor_list(self): return False

    def AddAndInferOp(self, op_conf):
        _AddAndInferMirroredOp(self.logical_blob_name, op_conf, self.sub_consistent_blob_list_)
        
    def _CheckNdarray(self, ndarray_list):
        assert isinstance(ndarray_list, (list, tuple))
        assert len(self.sub_consistent_blob_list_) == len(ndarray_list)
        def GetElemCnt(shape): return reduce(lambda x, y: x * y, shape, 1)
        for consistent_blob, ndarray in zip(self.sub_consistent_blob_list_, ndarray_list):
            assert type(ndarray) is np.ndarray
            assert len(ndarray.shape) == len(self.shape)
            assert GetElemCnt(ndarray.shape) <= GetElemCnt(self.shape)

    def _AsyncPush(self, session, ndarray_list):
        for i in range(len(ndarray_list)):
            sub_blob = self.sub_consistent_blob_list_[i]
            session.AsyncPush(sub_blob.op_name, _MakePushNdarrayCallback(ndarray_list[i]))
            
@oneflow_export('MirroredTensorListDef')
class MirroredTensorListDef(ArgBlobDef):
    def __init__(self, shape, dtype=data_type_util.kFloat, batch_axis=0, name=None):
        assert type(shape) is tuple
        assert type(batch_axis) is int
        if batch_axis < 0: batch_axis += len(shape)
        assert batch_axis >= 0
        assert batch_axis < len(shape)
        ArgBlobDef.__init__(self, shape, dtype=dtype, batch_axis=batch_axis, name=name)
        self.sub_consistent_blob_list_ = []
        
    @property
    def is_dynamic(self): return True

    @property
    def is_tensor_list(self): return True

    def AddAndInferOp(self, op_conf):
        _AddAndInferMirroredOp(self.logical_blob_name, op_conf, self.sub_consistent_blob_list_)
        
    def _CheckNdarray(self, ndarray_lists):
        assert isinstance(ndarray_lists, (list, tuple))
        assert len(self.sub_consistent_blob_list_) == len(ndarray_lists)
        def GetElemCnt(shape): return reduce(lambda x, y: x * y, shape, 1)
        for consistent_blob, ndarray_list in zip(self.sub_consistent_blob_list_, ndarray_lists):
            assert type(ndarray_list) is list
            elem_cnt = 0
            for ndarray in ndarray_list:
                assert type(ndarray) is np.ndarray
                assert len(ndarray.shape) == len(self.shape)
                elem_cnt += GetElemCnt(ndarray.shape)
            assert elem_cnt <= GetElemCnt(self.shape)

    def _AsyncPush(self, session, ndarray_lists):
        for i in range(len(ndarray_lists)):
            sub_blob = self.sub_consistent_blob_list_[i]
            session.AsyncPush(sub_blob.op_name, _MakePushNdarrayListCallback(ndarray_lists[i]))

def _AddAndInferMirroredOp(mirrored_lbn, op_conf, sub_consistent_blob_list):
    compile_context.CurJobAddMirroredOp(op_conf)
    job_name = g_func_ctx.JobBuildAndInferCtx_GetCurrentJobName()
    num_sub_lbi = g_func_ctx.JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, mirrored_lbn)
    for i in range(num_sub_lbi):
        sub_lbi = g_func_ctx.JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, mirrored_lbn, i)
        sub_consistent_blob_list.append(remote_blob_util.ConsistentBlob(sub_lbi))

def _MakePushNdarrayCallback(ndarray):
    copied = np.copy(ndarray)
    return lambda ofblob: ofblob.CopyFromNdarray(copied)

def _MakePushNdarrayListCallback(ndarray_list):
    copied = [np.copy(ndarray) for ndarray in ndarray_list]
    return lambda ofblob: ofblob.CopyFromNdarrayList(copied)

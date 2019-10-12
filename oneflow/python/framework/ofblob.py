from __future__ import absolute_import

import oneflow.core.common.data_type_pb2 as dtype_util
from oneflow.python.framework.dtype import convert_of_dtype_to_numpy_dtype
from oneflow.core.register.lod_tree_pb2 import LoDTree
import oneflow.oneflow_internal as oneflow_internal
import oneflow.python.framework.blob as blob_util
from google.protobuf import text_format
from oneflow.python.lib.core.box import Box
import numpy as np
import collections

class OfBlob(object):
    def __init__(self, of_blob_ptr):
        self.of_blob_ptr_ = of_blob_ptr

    @property
    def dtype(self):
        return oneflow_internal.Ofblob_GetDataType(self.of_blob_ptr_)

    @property
    def shape(self):
        num_axes = oneflow_internal.OfBlob_NumAxes(self.of_blob_ptr_)
        dst_ndarray = np.ndarray(num_axes, dtype=np.int64)
        oneflow_internal.OfBlob_CopyShapeToNumpy(self.of_blob_ptr_, dst_ndarray)
        return tuple(dst_ndarray.tolist())
    
    @property
    def is_dynamic(self):
        return oneflow_internal.OfBlob_IsDynamic(self.of_blob_ptr_)
    
    @property
    def num_of_lod_levels(self):
        num_of_lod_levels = oneflow_internal.OfBlob_GetNumOfLoDLevels(self.of_blob_ptr_)
        if num_of_lod_levels > 0:
            assert num_of_lod_levels > 1
        else:
            assert num_of_lod_levels == 0
        return num_of_lod_levels

    @property
    def lod_tree(self):
        lod_tree_str = oneflow_internal.OfBlob_GetSerializedLoDTree(self.of_blob_ptr_)
        return text_format.Parse(lod_tree_str, LoDTree())

    def CopyToBlob(self):
        blob = blob_util.Blob();
        dense_ndarray = self.CopyToNdarray()
        blob.set_ndarray(dense_ndarray)
        if self.num_of_lod_levels > 0:
            lod_tree = self.lod_tree
            blob.set_lod_tree(lod_tree)
            lod_ndarray_nested_list = self._MakeLoDNdarrayNestedList(lod_tree, dense_ndarray)
            blob.set_lod_ndarray_nested_list(lod_ndarray_nested_list)
        return blob

    def CopyToNdarray(self):
        dst_ndarray = np.ndarray(self._Size(), dtype=convert_of_dtype_to_numpy_dtype(self.dtype))
        method_name = oneflow_internal.Dtype_GetOfBlobCopyToBufferFuncName(self.dtype)
        copy_method = getattr(oneflow_internal, method_name)
        copy_method(self.of_blob_ptr_, dst_ndarray)
        return dst_ndarray.reshape(self.shape)

    def CopyFromNdarrayOrNestedNdarrayList(self, src_ndarray):
        if self.num_of_lod_levels > 0:
            assert isinstance(src_ndarray, (list, tuple))
            lod_tree, src_ndarray = self._MakeLodTreeAndDenseNdarray(src_ndarray)
            self._SetShape(np.array(src_ndarray.shape, dtype=np.int64))
            self._SetLoDTree(lod_tree)
        else:
            self._SetShape(np.array(src_ndarray.shape, dtype=np.int64))
        assert isinstance(src_ndarray, np.ndarray)
        self._CopyFromNdarray(src_ndarray)

    def _MakeLodTreeAndDenseNdarray(self, lod_ndarray_nested_list):
        lod_tree = LoDTree()
        shape = list(self.shape)
        shape[0] = 0
        offset = Box(0)
        blob_np_dtype = convert_of_dtype_to_numpy_dtype(self.dtype)
        dense_ndarray = Box(np.zeros(tuple(shape), dtype=blob_np_dtype))
        def RecursiveMakeLodTreeAndDenseNdarray(lod_tree, sub_ndarray_nested_list):
            if isinstance(sub_ndarray_nested_list, np.ndarray):
                assert sub_ndarray_nested_list.dtype == blob_np_dtype
                lod_tree.offset = offset.value
                lod_tree.length = len(sub_ndarray_nested_list)
                offset.set_value(offset.value + lod_tree.length)
                assert dense_ndarray.value.shape[1:] == sub_ndarray_nested_list.shape[1:],\
                    "lhs: %s, rhs: %s" %(dense_ndarray.value.shape[1:], \
                                         sub_ndarray_nested_list.shape[1:])
                dense_ndarray.set_value(np.concatenate((dense_ndarray.value, sub_ndarray_nested_list)))
            else:
                assert isinstance(sub_ndarray_nested_list, (list, tuple))
                idx = 0
                for x in sub_ndarray_nested_list:
                    sub_lod_tree = lod_tree.children.add()
                    RecursiveMakeLodTreeAndDenseNdarray(sub_lod_tree, x)
                    if idx == 0:
                        lod_tree.offset = sub_lod_tree.offset
                        lod_tree.length = 0
                    lod_tree.length += sub_lod_tree.length
                    idx += 1
        RecursiveMakeLodTreeAndDenseNdarray(lod_tree, lod_ndarray_nested_list)
        return lod_tree, dense_ndarray.value

    def _MakeLoDNdarrayNestedList(self, lod_tree, dense_ndarray):
        def RecursiveMakeLoDNdarrayNestedList(lod_tree):
            if len(lod_tree.children) == 0:
                start = lod_tree.offset
                end = start + lod_tree.length
                return dense_ndarray[start:end,]
            ndarray_list = []
            for x in lod_tree.children:
                ndarray_list.append(RecursiveMakeLoDNdarrayNestedList(x))
            return ndarray_list
        return RecursiveMakeLoDNdarrayNestedList(lod_tree)

    def _CopyFromNdarray(self, src_ndarray):
        of_dtype = convert_of_dtype_to_numpy_dtype(self.dtype)
        assert(of_dtype == src_ndarray.dtype), \
            "of_dtype: %s, numpy.dtype: %s" % (of_dtype, src_ndarray.dtype)
        method_name = oneflow_internal.Dtype_GetOfBlobCopyFromBufferFuncName(self.dtype)
        copy_method = getattr(oneflow_internal, method_name)
        copy_method(self.of_blob_ptr_, src_ndarray)
        return

    def _Size(self):
        elem_cnt = 1
        for d in self.shape:
            elem_cnt *= d
        return elem_cnt
    
    def _SetShape(self, shape_tensor):
        assert shape_tensor.dtype == np.int64
        assert len(shape_tensor) == oneflow_internal.OfBlob_NumAxes(self.of_blob_ptr_)
        oneflow_internal.OfBlob_CopyShapeFromNumpy(self.of_blob_ptr_, shape_tensor)
        
    def _SetLoDTree(self, lod_tree):
        lod_tree_str = str(text_format.MessageToString(lod_tree))
        oneflow_internal.OfBlob_SetSerializedLoDTree(self.of_blob_ptr_, lod_tree_str)

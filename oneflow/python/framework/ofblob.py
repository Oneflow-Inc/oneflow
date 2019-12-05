from __future__ import absolute_import

import oneflow.core.common.data_type_pb2 as dtype_util
from oneflow.python.framework.dtype import convert_of_dtype_to_numpy_dtype
from oneflow.core.register.lod_tree_pb2 import LoDTree
import oneflow.oneflow_internal as oneflow_api
import oneflow.python.framework.blob as blob_util
from google.protobuf import text_format
from oneflow.python.lib.core.box import Box
import numpy as np
import collections
from functools import reduce

class OfBlob(object):
    def __init__(self, of_blob_ptr):
        self.of_blob_ptr_ = of_blob_ptr

    @property
    def dtype(self):
        return oneflow_api.Ofblob_GetDataType(self.of_blob_ptr_)

    @property
    def shape(self):
        num_axes = oneflow_api.OfBlob_NumAxes(self.of_blob_ptr_)
        dst_ndarray = np.ndarray(num_axes, dtype=np.int64)
        oneflow_api.OfBlob_CopyShapeToNumpy(self.of_blob_ptr_, dst_ndarray)
        return tuple(dst_ndarray.tolist())
    
    @property
    def num_axes(self): return oneflow_api.OfBlob_NumAxes(self.of_blob_ptr_)

    @property
    def is_dynamic(self):
        return oneflow_api.OfBlob_IsDynamic(self.of_blob_ptr_)
    
    @property
    def num_of_lod_levels(self):
        num_of_lod_levels = oneflow_api.OfBlob_GetNumOfLoDLevels(self.of_blob_ptr_)
        if num_of_lod_levels > 0:
            assert num_of_lod_levels > 1
        else:
            assert num_of_lod_levels == 0
        return num_of_lod_levels

    @property
    def lod_tree(self):
        lod_tree_str = oneflow_api.OfBlob_GetSerializedLoDTree(self.of_blob_ptr_)
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
        ndarray_lists = self._CopyToNdarrayLists()
        assert len(ndarray_lists) == 1
        assert len(ndarray_lists[0]) == 1
        return ndarray_lists[0][0]

    def CopyFromNdarray(self, src_ndarray):
        return self._CopyFromNdarrayLists([[src_ndarray]])
   
    def CopyFromNdarrayOrNestedNdarrayList(self, src_ndarray):
        if self.num_of_lod_levels > 0:
            assert isinstance(src_ndarray, (list, tuple))
            lod_tree, src_ndarray = self._MakeLodTreeAndDenseNdarray(src_ndarray)
            self._SetShape(np.array(src_ndarray.shape, dtype=np.int64))
            self._SetLoDTree(lod_tree)
        else:
            self._SetShape(np.array(src_ndarray.shape, dtype=np.int64))
        assert isinstance(src_ndarray, np.ndarray)
        self.CopyFromNdarray(src_ndarray)

    def CopyToNdarrayLists(self):
        assert self.is_dynamic
        return self._CopyToNdarrayLists()

    def CopyFromNdarrayLists(self, ndarray_lists):
        assert self.is_dynamic
        return self._CopyFromNdarrayLists(ndarray_lists)
    
    def _CopyToNdarrayLists(self):
        tensor_list, is_new_slice_start_mask = self._CopyToNdarrayListAndIsNewSliceStartMask()
        tensor_lists = []
        for tensor, is_new_slice_start in zip(tensor_list, is_new_slice_start_mask):
            if is_new_slice_start: tensor_lists.append([])
            tensor_lists[-1].append(tensor)
        return tensor_lists

    def _CopyToNdarrayListAndIsNewSliceStartMask(self):
        # get tensor list
        method_name = oneflow_api.Dtype_GetOfBlobCurTensorCopyToBufferFuncName(self.dtype)
        copy_method = getattr(oneflow_api, method_name)
        tensor_list = []
        oneflow_api.OfBlob_ResetTensorIterator(self.of_blob_ptr_)
        while oneflow_api.OfBlob_CurTensorIteratorEqEnd(self.of_blob_ptr_) == False:
            shape_tensor = np.ndarray(self.num_axes, dtype=np.int64)
            oneflow_api.OfBlob_CurTensorCopyShapeTo(self.of_blob_ptr_, shape_tensor)
            shape = tuple(shape_tensor.tolist())
            tensor = np.ndarray(shape, dtype=convert_of_dtype_to_numpy_dtype(self.dtype))
            copy_method(self.of_blob_ptr_, tensor)
            tensor_list.append(tensor)
            oneflow_api.OfBlob_IncTensorIterator(self.of_blob_ptr_)
        assert len(tensor_list) == oneflow_api.OfBlob_TotalNumOfTensors(self.of_blob_ptr_)
        # generate is_new_slice_start_mask
        is_new_slice_start_mask = [False] * len(tensor_list)
        num_slices = oneflow_api.OfBlob_NumOfTensorListSlices(self.of_blob_ptr_)
        for x in range(num_slices):
            tensor_list_start = oneflow_api.OfBlob_TensorIndex4SliceId(self.of_blob_ptr_, x)
            assert tensor_list_start >= 0
            assert tensor_list_start < len(is_new_slice_start_mask)
            is_new_slice_start_mask[tensor_list_start] = True
        return tensor_list, is_new_slice_start_mask
    
    def _CopyFromNdarrayLists(self, ndarray_lists):
        assert isinstance(ndarray_lists, (list, tuple))
        flat_ndarray_list = []
        is_new_slice_start_mask = []
        for ndarray_list in ndarray_lists:
            assert isinstance(ndarray_list, (list, tuple))
            for i, ndarray in enumerate(ndarray_list):
                assert isinstance(ndarray, np.ndarray)
                flat_ndarray_list.append(ndarray)
                is_new_slice_start_mask.append(i == 0)
        self._CopyFromNdarrayListAndIsNewSliceStartMask(flat_ndarray_list, is_new_slice_start_mask) 

    def _CopyFromNdarrayListAndIsNewSliceStartMask(self, tensor_list, is_new_slice_start_mask):
        assert len(tensor_list) == len(is_new_slice_start_mask)
        method_name = oneflow_api.Dtype_GetOfBlobCurMutTensorCopyFromBufferFuncName(self.dtype)
        copy_method = getattr(oneflow_api, method_name)
        oneflow_api.OfBlob_ClearTensorLists(self.of_blob_ptr_)
        for i, tensor in enumerate(tensor_list):
            if is_new_slice_start_mask[i]: oneflow_api.OfBlob_AddTensorListSlice(self.of_blob_ptr_)
            oneflow_api.OfBlob_AddTensor(self.of_blob_ptr_)
            assert oneflow_api.OfBlob_CurMutTensorIsNull(self.of_blob_ptr_) == False
            shape_tensor = np.array(tensor.shape, dtype=np.int64)
            oneflow_api.OfBlob_CurMutTensorCopyShapeFrom(self.of_blob_ptr_, shape_tensor)
            copy_method(self.of_blob_ptr_, tensor)
        assert len(tensor_list) == oneflow_api.OfBlob_TotalNumOfTensors(self.of_blob_ptr_)
        num_slices = reduce(lambda a, b: a + b, is_new_slice_start_mask, 0)
        assert num_slices == oneflow_api.OfBlob_NumOfTensorListSlices(self.of_blob_ptr_)

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

    def _SetShape(self, shape_tensor):
        assert shape_tensor.dtype == np.int64
        assert len(shape_tensor) == oneflow_api.OfBlob_NumAxes(self.of_blob_ptr_)
        oneflow_api.OfBlob_CopyShapeFromNumpy(self.of_blob_ptr_, shape_tensor)
        
    def _SetLoDTree(self, lod_tree):
        lod_tree_str = str(text_format.MessageToString(lod_tree))
        oneflow_api.OfBlob_SetSerializedLoDTree(self.of_blob_ptr_, lod_tree_str)

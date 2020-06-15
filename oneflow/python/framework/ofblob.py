from __future__ import absolute_import

import collections
from functools import reduce

import numpy as np
import oneflow.core.common.data_type_pb2 as dtype_util
import oneflow.oneflow_internal as oneflow_api
import oneflow.python.framework.local_blob as local_blob_util
import oneflow.python.framework.remote_blob as remote_blob_util
from google.protobuf import text_format
from oneflow.python.framework.dtype import convert_of_dtype_to_numpy_dtype
from oneflow.python.lib.core.box import Box


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
    def num_axes(self):
        return oneflow_api.OfBlob_NumAxes(self.of_blob_ptr_)

    @property
    def is_dynamic(self):
        return oneflow_api.OfBlob_IsDynamic(self.of_blob_ptr_)

    @property
    def is_tensor_list(self):
        return oneflow_api.OfBlob_IsTensorList(self.of_blob_ptr_)

    def CopyToNdarray(self):
        ndarray_lists = self._CopyToNdarrayLists()
        assert len(ndarray_lists) == 1
        assert len(ndarray_lists[0]) == 1
        return ndarray_lists[0][0]

    def CopyToNdarrayLists(self):
        return self._CopyToNdarrayLists()

    def CopyFromNdarray(self, src_ndarray):
        return self._CopyFromNdarrayLists([[src_ndarray]])

    def CopyFromNdarrayList(self, src_ndarray_list):
        return self._CopyFromNdarrayLists([src_ndarray_list])

    def CopyFromNdarrayLists(self, ndarray_lists):
        assert self.is_dynamic
        return self._CopyFromNdarrayLists(ndarray_lists)

    def _CopyToNdarrayLists(self):
        (
            tensor_list,
            is_new_slice_start_mask,
        ) = self._CopyToNdarrayListAndIsNewSliceStartMask()
        tensor_lists = []
        for tensor, is_new_slice_start in zip(tensor_list, is_new_slice_start_mask):
            if is_new_slice_start:
                tensor_lists.append([])
            tensor_lists[-1].append(tensor)
        return tensor_lists

    def _CopyToNdarrayListAndIsNewSliceStartMask(self):
        # get tensor list
        method_name = oneflow_api.Dtype_GetOfBlobCurTensorCopyToBufferFuncName(
            self.dtype
        )
        copy_method = getattr(oneflow_api, method_name)
        tensor_list = []
        oneflow_api.OfBlob_ResetTensorIterator(self.of_blob_ptr_)
        while oneflow_api.OfBlob_CurTensorIteratorEqEnd(self.of_blob_ptr_) == False:
            shape_tensor = np.ndarray(self.num_axes, dtype=np.int64)
            oneflow_api.OfBlob_CurTensorCopyShapeTo(self.of_blob_ptr_, shape_tensor)
            shape = tuple(shape_tensor.tolist())
            tensor = np.ndarray(
                shape, dtype=convert_of_dtype_to_numpy_dtype(self.dtype)
            )
            copy_method(self.of_blob_ptr_, tensor)
            tensor_list.append(tensor)
            oneflow_api.OfBlob_IncTensorIterator(self.of_blob_ptr_)
        assert len(tensor_list) == oneflow_api.OfBlob_TotalNumOfTensors(
            self.of_blob_ptr_
        )
        # generate is_new_slice_start_mask
        is_new_slice_start_mask = [False] * len(tensor_list)
        num_slices = oneflow_api.OfBlob_NumOfTensorListSlices(self.of_blob_ptr_)
        for x in range(num_slices):
            tensor_list_start = oneflow_api.OfBlob_TensorIndex4SliceId(
                self.of_blob_ptr_, x
            )
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
        self._CopyFromNdarrayListAndIsNewSliceStartMask(
            flat_ndarray_list, is_new_slice_start_mask
        )

    def _CopyFromNdarrayListAndIsNewSliceStartMask(
        self, tensor_list, is_new_slice_start_mask
    ):
        assert len(tensor_list) == len(is_new_slice_start_mask)
        method_name = oneflow_api.Dtype_GetOfBlobCurMutTensorCopyFromBufferFuncName(
            self.dtype
        )
        copy_method = getattr(oneflow_api, method_name)
        oneflow_api.OfBlob_ClearTensorLists(self.of_blob_ptr_)
        for i, tensor in enumerate(tensor_list):
            if is_new_slice_start_mask[i]:
                oneflow_api.OfBlob_AddTensorListSlice(self.of_blob_ptr_)
            oneflow_api.OfBlob_AddTensor(self.of_blob_ptr_)
            assert oneflow_api.OfBlob_CurMutTensorAvailable(self.of_blob_ptr_)
            shape_tensor = np.array(tensor.shape, dtype=np.int64)
            oneflow_api.OfBlob_CurMutTensorCopyShapeFrom(
                self.of_blob_ptr_, shape_tensor
            )
            copy_method(self.of_blob_ptr_, tensor)
        assert len(tensor_list) == oneflow_api.OfBlob_TotalNumOfTensors(
            self.of_blob_ptr_
        )
        num_slices = reduce(lambda a, b: a + b, is_new_slice_start_mask, 0)
        assert num_slices == oneflow_api.OfBlob_NumOfTensorListSlices(self.of_blob_ptr_)

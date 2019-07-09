from __future__ import absolute_import

import oneflow_internal
from oneflow.python.framework.dtype import convert_of_dtype_to_numpy_dtype
import numpy as np

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

    def CopyToNdarray(self):
        dst_ndarray = np.ndarray(self._size, dtype=convert_of_dtype_to_numpy_dtype(self.dtype))
        method_name = oneflow_internal.Dtype_GetOfBlobCopyToBufferFuncName(self.dtype)
        copy_method = getattr(oneflow_internal, method_name)
        copy_method(self.of_blob_ptr_, dst_ndarray)
        return dst_ndarray.reshape(self.shape)

    def CopyFromNdarray(self, src_ndarray):
        assert(self._size == src_ndarray.size)
        assert(convert_of_dtype_to_numpy_dtype(self.dtype) == src_ndarray.dtype)
        method_name = oneflow_internal.Dtype_GetOfBlobCopyFromBufferFuncName(self.dtype)
        copy_method = getattr(oneflow_internal, method_name)
        copy_method(self.of_blob_ptr_, src_ndarray)
        return

    @property
    def _size(self):
        elem_cnt = 1
        for d in self.shape:
            elem_cnt *= d
        return elem_cnt

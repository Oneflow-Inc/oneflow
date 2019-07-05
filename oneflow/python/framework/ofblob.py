import oneflow_internal
import oneflow.core.common.data_type_pb2 as data_type_pb2
import numpy as np

OF_BLOB_DTYPE2NUMPY_DTYPE = {
        data_type_pb2.kInt32: np.int32,
        data_type_pb2.kFloat: np.float32
    }

def convert_of_dtype_to_numpy_dtype(dtype):
    if dtype not in OF_BLOB_DTYPE2NUMPY_DTYPE: raise NotImplementedError
    return OF_BLOB_DTYPE2NUMPY_DTYPE[dtype ]

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
        return dst_ndarray.tolist()

    def numpy(self):
        dst_ndarray = np.ndarray(self._Size, dtype=convert_of_dtype_to_numpy_dtype(self.dtype))
        method_name = oneflow_internal.OfBlob_GetCopyToBufferFuncName(self.of_blob_ptr_)
        copy_method = getattr(oneflow_internal, method_name)
        copy_method(self.of_blob_ptr_, dst_ndarray)
        return dst_ndarray.reshape(self.shape)

    def copyFromNumpy(self, src_ndarray):
        assert(self._Size == src_ndarray.size)
        assert(convert_of_dtype_to_numpy_dtype(self.dtype) == src_ndarray.dtype)
        method_name = oneflow_internal.OfBlob_GetCopyFromBufferFuncName(self.of_blob_ptr_)
        copy_method = getattr(oneflow_internal, method_name)
        copy_method(self.of_blob_ptr_, src_ndarray)
        return

    @property
    def _Size(self):
        elem_cnt = 1
        for d in self.shape:
            elem_cnt *= d
        return elem_cnt

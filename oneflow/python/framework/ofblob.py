import oneflow_internal
import oneflow.core.common.data_type_pb2 as data_type_pb2
import numpy as np

OF_BLOB_DTYPE2NUMPY_DTYPE = {
        data_type_pb2.kInt32: np.int32,
        data_type_pb2.kFloat: np.float32
    }

def dtypeInNumpy(dtype):
    ret = OF_BLOB_DTYPE2NUMPY_DTYPE.get(dtype)
    if ret is None:
        raise NotImplementedError
    else:
        return ret

class OfBlob(object):
    def __init__(self, of_blob_ptr):
        self.of_blob_ptr_ = of_blob_ptr

    @property
    def dtype(self):
        return oneflow_internal.Ofblob_GetDataType(self.of_blob_ptr_)

    @property
    def shape(self):
        num_axes = oneflow_internal.OfBlob_NumAxes(self.of_blob_ptr_)
        dst_ndarray = np.ndarray(num_axes, dtype=np.int)
        oneflow_internal.OfBlob_CopyShapeToNumpy(dst_ndarray, self.of_blob_ptr_)
        return dst_ndarray.tolist()

    def numpy(self):
        elem_cnt = 1
        for d in self.shape:
            elem_cnt *= d
        dtype_ = dtypeInNumpy(self.dtype)
        dst_ndarray = np.ndarray(elem_cnt, dtype=dtype_)
        if dtype_ is np.int32:
            oneflow_internal.CopyToInt32Ndarry(dst_ndarray, self.of_blob_ptr_)
        elif dtype_ is np.float32:
            oneflow_internal.CopyToFloat32Ndarry(dst_ndarray, self.of_blob_ptr_)
        else:
            raise NotImplementedError
        return dst_ndarray

    def copyFromNumpy(self, src_ndarray):
        elem_cnt = 1
        for d in self.shape:
            elem_cnt *= d
        assert(elem_cnt == src_ndarray.size)
        dtype_ = dtypeInNumpy(self.dtype)
        if dtype_ is np.int32:
            oneflow_internal.CopyFromInt32Ndarry(src_ndarray, self.of_blob_ptr_)
        elif dtype_ is np.float32:
            oneflow_internal.CopyFromFloat32Ndarry(src_ndarray, self.of_blob_ptr_)
        else:
            raise NotImplementedError
        return

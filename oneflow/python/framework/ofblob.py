import oneflow_internal
import oneflow.core.common.data_type_pb2 as data_type_pb2
import numpy as np

class OfBlob(object):
    def __init__(self, of_blob_ptr):
        self.of_blob_ptr_ = of_blob_ptr

    @property
    def dtype(self):
        return oneflow_internal.Ofblob_GetDataType(self.of_blob_ptr_)
        
    @property
    def dtypeInNumpy(self):
        if self.dtype is data_type_pb2.kInt32:
            return np.int32
        elif self.dtype is data_type_pb2.kFloat:
            return np.float32
        else:
            raise NotImplementedError

    @property
    def shape(self):
        TODO()

    @property
    def elemCnt(self):
        return oneflow_internal.Ofblob_GetElemCnt(self.of_blob_ptr_)

    def numpy(self):
        dst_ndarray = np.ndarray(self.elemCnt, dtype=self.dtypeInNumpy)
        if self.dtypeInNumpy is np.int32:
            oneflow_internal.CopyToInt32Ndarry(dst_ndarray, self.of_blob_ptr_)
        elif self.dtypeInNumpy is np.float32:
            oneflow_internal.CopyToFloat32Ndarry(dst_ndarray, self.of_blob_ptr_)
        else:
            raise NotImplementedError
        return dst_ndarray

    def copyFromNumpy(self, src_ndarray):
        assert(self.elemCnt == src_ndarray.size)
        dtype_ = self.dtypeInNumpy()
        elem_cnt = oneflow_internal.Ofblob_GetElemCnt(self.of_blob_ptr_)
        if self.dtypeInNumpy is np.int32:
            oneflow_internal.CopyFromInt32Ndarry(src_ndarray, self.of_blob_ptr_)
        elif self.dtypeInNumpy is np.float32:
            oneflow_internal.CopyFromFloat32Ndarry(src_ndarray, self.of_blob_ptr_)
        else:
            raise NotImplementedError
        return

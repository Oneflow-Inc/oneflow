from __future__ import absolute_import
import oneflow.core.common.data_type_pb2 as data_type_pb2
import numpy as np

char = data_type_pb2.kChar
float = data_type_pb2.kFloat
float32 = float
double = data_type_pb2.kDouble
float64 = double
int8 = data_type_pb2.kInt8
int32 = data_type_pb2.kInt32
int64 = data_type_pb2.kInt64
uint8 = data_type_pb2.kUInt8

_OF_BLOB_DTYPE2NUMPY_DTYPE = {
        data_type_pb2.kInt8: np.int8,
        data_type_pb2.kInt32: np.int32,
        data_type_pb2.kInt64: np.int64,
        data_type_pb2.kUInt8: np.uint8,
        data_type_pb2.kFloat: np.float32,
        data_type_pb2.kDouble: np.double,
        # could be np.ubyte on some platform
        data_type_pb2.kChar: np.byte, 
    }

def convert_of_dtype_to_numpy_dtype(dtype):
    if dtype not in _OF_BLOB_DTYPE2NUMPY_DTYPE: raise NotImplementedError
    return _OF_BLOB_DTYPE2NUMPY_DTYPE[dtype]

del absolute_import
del data_type_pb2
del np

from __future__ import absolute_import

import numpy as np
import oneflow.core.common.data_type_pb2 as data_type_pb2


class Dtype(object):
    def __init__(self, oneflow_dtype, numpy_dtype):
        self.oneflow_dtype = oneflow_dtype
        self.numpy_dtype = numpy_dtype


class Char(Dtype):
    def __init__(self, oneflow_dtype, numpy_dtype):
        Dtype.__init__(self, oneflow_dtype, numpy_dtype)


class Float(Dtype):
    def __init__(self, oneflow_dtype, numpy_dtype):
        Dtype.__init__(self, oneflow_dtype, numpy_dtype)


class Float16(Dtype):
    def __init__(self, oneflow_dtype, numpy_dtype):
        Dtype.__init__(self, oneflow_dtype, numpy_dtype)


class Float32(Dtype):
    def __init__(self, oneflow_dtype, numpy_dtype):
        Dtype.__init__(self, oneflow_dtype, numpy_dtype)


class Float64(Dtype):
    def __init__(self, oneflow_dtype, numpy_dtype):
        Dtype.__init__(self, oneflow_dtype, numpy_dtype)


class Double(Dtype):
    def __init__(self, oneflow_dtype, numpy_dtype):
        Dtype.__init__(self, oneflow_dtype, numpy_dtype)


class Int8(Dtype):
    def __init__(self, oneflow_dtype, numpy_dtype):
        Dtype.__init__(self, oneflow_dtype, numpy_dtype)


class Int32(Dtype):
    def __init__(self, oneflow_dtype, numpy_dtype):
        Dtype.__init__(self, oneflow_dtype, numpy_dtype)


class Int64(Dtype):
    def __init__(self, oneflow_dtype, numpy_dtype):
        Dtype.__init__(self, oneflow_dtype, numpy_dtype)


class Uint8(Dtype):
    def __init__(self, oneflow_dtype, numpy_dtype):
        Dtype.__init__(self, oneflow_dtype, numpy_dtype)


char = Char(data_type_pb2.kChar, np.byte)
float = Float(data_type_pb2.kFloat, np.float)
float16 = Float16(data_type_pb2.kFloat16, np.float16)
float32 = Float32(data_type_pb2.kFloat, np.float32)
float64 = Float64(data_type_pb2.kDouble, np.double)
double = Double(data_type_pb2.kDouble, np.double)
int8 = Int8(data_type_pb2.kInt8, np.int8)
int32 = Int32(data_type_pb2.kInt32, np.int32)
int64 = Int64(data_type_pb2.kInt64, np.int64)
uint8 = Uint8(data_type_pb2.kUInt8, np.uint8)


dtypes = [
    char,
    float,
    float32,
    double,
    float64,
    float16,
    int8,
    int32,
    int64,
    uint8,
]

_OF_BLOB_DTYPE2ONEFLOW_DTYPE_CLASS = {
    data_type_pb2.kInt8: int8,
    data_type_pb2.kInt32: int32,
    data_type_pb2.kInt64: int64,
    data_type_pb2.kUInt8: uint8,
    data_type_pb2.kFloat: float32,
    data_type_pb2.kDouble: double,
    data_type_pb2.kFloat16: float16,
    # could be np.ubyte on some platform
    data_type_pb2.kChar: char,
}


def convert_of_dtype_to_oneflow_dtype_class(dtype):
    if dtype not in _OF_BLOB_DTYPE2ONEFLOW_DTYPE_CLASS:
        raise NotImplementedError
    return _OF_BLOB_DTYPE2ONEFLOW_DTYPE_CLASS[dtype]


del absolute_import
del data_type_pb2
del np

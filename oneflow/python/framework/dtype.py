from __future__ import absolute_import

import numpy as np
import oneflow.core.common.data_type_pb2 as data_type_pb2


class dtype(object):
    oneflow_dtype = data_type_pb2.kInvalidDataType
    numpy_dtype = data_type_pb2.kInvalidDataType


class char(dtype):
    oneflow_dtype = data_type_pb2.kChar
    numpy_dtype = np.byte


class float(dtype):
    oneflow_dtype = data_type_pb2.kFloat
    numpy_dtype = np.float


class float16(dtype):
    oneflow_dtype = data_type_pb2.kFloat16
    numpy_dtype = np.float16


class float32(dtype):
    oneflow_dtype = data_type_pb2.kFloat
    numpy_dtype = np.float32


class float64(dtype):
    oneflow_dtype = data_type_pb2.kDouble
    numpy_dtype = np.double


class double(dtype):
    oneflow_dtype = data_type_pb2.kDouble
    numpy_dtype = np.double


class int8(dtype):
    oneflow_dtype = data_type_pb2.kInt8
    numpy_dtype = np.int8


class int32(dtype):
    oneflow_dtype = data_type_pb2.kInt32
    numpy_dtype = np.int32


class int64(dtype):
    oneflow_dtype = data_type_pb2.kInt64
    numpy_dtype = np.int64


class uint8(dtype):
    oneflow_dtype = data_type_pb2.kUInt8
    numpy_dtype = np.uint8


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

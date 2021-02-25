"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import

import numpy as np
import oneflow.core.common.data_type_pb2 as data_type_pb2
from oneflow.python.oneflow_export import oneflow_export


class dtype(object):
    oneflow_proto_dtype = data_type_pb2.kInvalidDataType


@oneflow_export("char")
class char(dtype):
    oneflow_proto_dtype = data_type_pb2.kChar


@oneflow_export("float16")
class float16(dtype):
    oneflow_proto_dtype = data_type_pb2.kFloat16


@oneflow_export("float")
@oneflow_export("float32")
class float32(dtype):
    oneflow_proto_dtype = data_type_pb2.kFloat


float = float32


@oneflow_export("double")
@oneflow_export("float64")
class float64(dtype):
    oneflow_proto_dtype = data_type_pb2.kDouble


double = float64


@oneflow_export("int8")
class int8(dtype):
    oneflow_proto_dtype = data_type_pb2.kInt8


@oneflow_export("int32")
class int32(dtype):
    oneflow_proto_dtype = data_type_pb2.kInt32


@oneflow_export("int64")
class int64(dtype):
    oneflow_proto_dtype = data_type_pb2.kInt64


@oneflow_export("uint8")
class uint8(dtype):
    oneflow_proto_dtype = data_type_pb2.kUInt8


@oneflow_export("record")
class record(dtype):
    oneflow_proto_dtype = data_type_pb2.kOFRecord


@oneflow_export("tensor_buffer")
class tensor_buffer(dtype):
    oneflow_proto_dtype = data_type_pb2.kTensorBuffer


_dtypes = [
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
    record,
    tensor_buffer,
]


@oneflow_export("dtypes")
def dtypes():
    return _dtypes


_PROTO_DTYPE2ONEFLOW_DTYPE = {
    data_type_pb2.kInt8: int8,
    data_type_pb2.kInt32: int32,
    data_type_pb2.kInt64: int64,
    data_type_pb2.kUInt8: uint8,
    data_type_pb2.kFloat: float32,
    data_type_pb2.kDouble: double,
    data_type_pb2.kFloat16: float16,
    data_type_pb2.kChar: char,
    data_type_pb2.kOFRecord: record,
    data_type_pb2.kTensorBuffer: tensor_buffer,
}


def convert_proto_dtype_to_oneflow_dtype(proto_dtype):
    if proto_dtype not in _PROTO_DTYPE2ONEFLOW_DTYPE:
        raise NotImplementedError("proto_dtype %s not found in dict" % proto_dtype)
    return _PROTO_DTYPE2ONEFLOW_DTYPE[proto_dtype]


_ONEFLOW_DTYPE_TO_NUMPY_DTYPE = {
    # could be np.ubyte on some platform
    char: np.byte,
    float: np.float32,
    float16: np.float16,
    float32: np.float32,
    float64: np.double,
    double: np.double,
    int8: np.int8,
    int32: np.int32,
    int64: np.int64,
    uint8: np.uint8,
}


@oneflow_export("convert_oneflow_dtype_to_numpy_dtype")
def convert_oneflow_dtype_to_numpy_dtype(oneflow_dtype: dtype):
    if oneflow_dtype not in _ONEFLOW_DTYPE_TO_NUMPY_DTYPE:
        raise NotImplementedError
    return _ONEFLOW_DTYPE_TO_NUMPY_DTYPE[oneflow_dtype]


def convert_numpy_dtype_to_oneflow_dtype(numpy_dtype):
    for k, v in _ONEFLOW_DTYPE_TO_NUMPY_DTYPE.items():
        if v == numpy_dtype:
            return k
    raise NotImplementedError


_ONEFLOW_DTYPE_TO_PROTO_DTYPE = {
    int8: data_type_pb2.kInt8,
    int32: data_type_pb2.kInt32,
    int64: data_type_pb2.kInt64,
    uint8: data_type_pb2.kUInt8,
    float32: data_type_pb2.kFloat,
    double: data_type_pb2.kDouble,
    float16: data_type_pb2.kFloat16,
    char: data_type_pb2.kChar,
    record: data_type_pb2.kOFRecord,
    tensor_buffer: data_type_pb2.kTensorBuffer,
}


def convert_oneflow_dtype_to_proto_dtype(oneflow_dtype: dtype):
    if oneflow_dtype not in _ONEFLOW_DTYPE_TO_PROTO_DTYPE:
        raise NotImplementedError
    return _ONEFLOW_DTYPE2PROTO_DTYPE[oneflow_dtype]


del data_type_pb2
del np

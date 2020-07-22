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

char = data_type_pb2.kChar
float = data_type_pb2.kFloat
float32 = float
double = data_type_pb2.kDouble
float64 = double
float16 = data_type_pb2.kFloat16
int8 = data_type_pb2.kInt8
int32 = data_type_pb2.kInt32
int64 = data_type_pb2.kInt64
uint8 = data_type_pb2.kUInt8

dtypes = [char, float, float32, double, float64, float16, int8, int32, int64, uint8]

_OF_BLOB_DTYPE2NUMPY_DTYPE = {
    data_type_pb2.kInt8: np.int8,
    data_type_pb2.kInt32: np.int32,
    data_type_pb2.kInt64: np.int64,
    data_type_pb2.kUInt8: np.uint8,
    data_type_pb2.kFloat: np.float32,
    data_type_pb2.kDouble: np.double,
    data_type_pb2.kFloat16: np.float16,
    # could be np.ubyte on some platform
    data_type_pb2.kChar: np.byte,
}


def convert_of_dtype_to_numpy_dtype(dtype):
    if dtype not in _OF_BLOB_DTYPE2NUMPY_DTYPE:
        raise NotImplementedError
    return _OF_BLOB_DTYPE2NUMPY_DTYPE[dtype]


del absolute_import
del data_type_pb2
del np

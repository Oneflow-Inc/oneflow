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
import numpy as np

import oneflow._oneflow_internal
from oneflow.compatible import single_client as flow
from oneflow.core.common import data_type_pb2 as data_type_pb2

_dtypes = [
    flow.bool,
    flow.char,
    flow.float,
    flow.float32,
    flow.double,
    flow.float64,
    flow.float16,
    flow.int8,
    flow.int32,
    flow.int64,
    flow.uint8,
    flow.record,
    flow.tensor_buffer,
    flow.bfloat16,
    flow.uint16,
    flow.uint32,
    flow.uint64,
    flow.uint128,
    flow.int16,
    flow.int64,
    flow.int128,
    flow.complex32,
    flow.complex64,
    flow.complex128,
]


def dtypes():
    return _dtypes


def convert_proto_dtype_to_oneflow_dtype(proto_dtype):
    return oneflow._oneflow_internal.deprecated.GetDTypeByDataType(proto_dtype)


_ONEFLOW_DTYPE_TO_NUMPY_DTYPE = {
    flow.bool: np.bool,
    flow.char: np.byte,
    flow.float: np.float32,
    flow.float16: np.float16,
    flow.float32: np.float32,
    flow.float64: np.double,
    flow.double: np.double,
    flow.int8: np.int8,
    flow.int32: np.int32,
    flow.int64: np.int64,
    flow.uint8: np.uint8,
    flow.uint16: np.uint16,
    flow.uint32: np.uint32,
    flow.uint64: np.uint64,
    flow.int16: np.int16,
    flow.int64: np.int64,
    flow.complex64: np.complex64,
    flow.complex128: np.complex128,
}


def convert_oneflow_dtype_to_numpy_dtype(oneflow_dtype: flow.dtype):
    if oneflow_dtype not in _ONEFLOW_DTYPE_TO_NUMPY_DTYPE:
        raise NotImplementedError
    return _ONEFLOW_DTYPE_TO_NUMPY_DTYPE[oneflow_dtype]


def convert_numpy_dtype_to_oneflow_dtype(numpy_dtype: np.dtype):
    for (k, v) in _ONEFLOW_DTYPE_TO_NUMPY_DTYPE.items():
        if v == numpy_dtype:
            return k
    raise NotImplementedError


del data_type_pb2
del np

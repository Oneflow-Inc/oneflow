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
import oneflow
import oneflow_api


_dtypes = [
    oneflow.char,
    oneflow.float,
    oneflow.float32,
    oneflow.double,
    oneflow.float64,
    oneflow.float16,
    oneflow.int8,
    oneflow.int32,
    oneflow.int64,
    oneflow.uint8,
    oneflow.record,
    oneflow.tensor_buffer,
]


@oneflow_export("dtypes")
def dtypes():
    return _dtypes


def convert_proto_dtype_to_oneflow_dtype(proto_dtype):
    return oneflow_api.deprecated.GetDTypeByDataType(proto_dtype)


_ONEFLOW_DTYPE_TO_NUMPY_DTYPE = {
    # could be np.ubyte on some platform
    oneflow.char: np.byte,
    oneflow.float: np.float32,
    oneflow.float16: np.float16,
    oneflow.float32: np.float32,
    oneflow.float64: np.double,
    oneflow.double: np.double,
    oneflow.int8: np.int8,
    oneflow.int32: np.int32,
    oneflow.int64: np.int64,
    oneflow.uint8: np.uint8,
}


@oneflow_export("convert_oneflow_dtype_to_numpy_dtype")
def convert_oneflow_dtype_to_numpy_dtype(oneflow_dtype: oneflow.dtype):
    if oneflow_dtype not in _ONEFLOW_DTYPE_TO_NUMPY_DTYPE:
        raise NotImplementedError
    return _ONEFLOW_DTYPE_TO_NUMPY_DTYPE[oneflow_dtype]


def convert_numpy_dtype_to_oneflow_dtype(numpy_dtype):
    for k, v in _ONEFLOW_DTYPE_TO_NUMPY_DTYPE.items():
        if v == numpy_dtype:
            return k
    raise NotImplementedError


del data_type_pb2
del np

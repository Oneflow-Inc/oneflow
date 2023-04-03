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

import oneflow
import oneflow._oneflow_internal
import oneflow.core.common.data_type_pb2 as data_type_pb2
from oneflow._oneflow_internal import (
    set_default_dtype,
    get_default_dtype,
)

_dtypes = [
    oneflow.bool,
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
    oneflow.bfloat16,
    oneflow.complex64,
    oneflow.cfloat,
    oneflow.complex128,
    oneflow.cdouble,
]


def dtypes():
    return _dtypes


def convert_proto_dtype_to_oneflow_dtype(proto_dtype):
    return oneflow._oneflow_internal.deprecated.GetDTypeByDataType(proto_dtype)


_ONEFLOW_DTYPE_TO_NUMPY_DTYPE = {
    # >> np_bool = np.array([1,2], dtype=bool).dtype
    # >> np_bool == bool
    # True
    oneflow.bool: bool,
    oneflow.float: np.float32,
    oneflow.float16: np.float16,
    oneflow.float32: np.float32,
    oneflow.float64: np.double,
    oneflow.double: np.double,
    oneflow.int8: np.int8,
    oneflow.int32: np.int32,
    oneflow.int64: np.int64,
    oneflow.uint8: np.uint8,
    oneflow.complex64: np.complex64,
    oneflow.cfloat: np.complex64,
    oneflow.complex128: np.complex128,
    oneflow.cdouble: np.complex128,
}


def convert_oneflow_dtype_to_numpy_dtype(oneflow_dtype: oneflow.dtype):
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


def set_default_tensor_type(tensor_type):
    """Sets the default floating point type for those source operators which create Tensor.

    The default floating point type is ``oneflow.FloatTensor``.

    Args:
        tensor_type (type or string): The floating point tensor type or its name.

    For example:

    .. code-block:: python

        >>> import oneflow
        >>> oneflow.set_default_tensor_type(oneflow.FloatTensor)
        >>> x = oneflow.ones(2, 3)
        >>> x.dtype
        oneflow.float32
        >>> oneflow.set_default_tensor_type("oneflow.DoubleTensor")
        >>> x = oneflow.ones(2, 3)
        >>> x.dtype
        oneflow.float64
        >>> oneflow.set_default_tensor_type(oneflow.FloatTensor)
        >>> x = oneflow.tensor([1.0, 2])
        >>> x.dtype
        oneflow.float32
    """

    def _import_dotted_name(name):
        """
        This function quotes from: https://github.com/pytorch/pytorch/blob/master/torch/_utils.py
        """
        components = name.split(".")
        obj = __import__(components[0])
        for component in components[1:]:
            obj = getattr(obj, component)
        return obj

    if isinstance(tensor_type, str):
        tensor_type = _import_dotted_name(tensor_type)
    oneflow._oneflow_internal.set_default_tensor_type(tensor_type)


def is_floating_point(input):
    return input.is_floating_point()

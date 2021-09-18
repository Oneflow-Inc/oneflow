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
import oneflow as flow
from oneflow.framework.tensor import register_tensor_op


_ordered_dtype_list = [flow.int8, flow.int32, flow.int64, flow.float32, flow.float64]


def _infered_binary_op_dtype(dtype_a, dtype_b):
    """
    Infer the highest hierarchy dtype. 
    """
    inferred_dtype = max(dtype_a, dtype_b, key=_ordered_dtype_list.index)
    return inferred_dtype


@register_tensor_op("eq")
def eq_op(input, other):
    """
    Computes element-wise equality.
    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.

    Args:
        input (oneflow.Tensor): the tensor to compare
        other (oneflow.Tensor, float or int): the target to compare

    Returns:

        - A boolean tensor that is True where :attr:`input` is equal to :attr:`other` and False elsewhere

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.tensor(np.array([2, 3, 4, 5]), dtype=flow.float32)
        >>> other = flow.tensor(np.array([2, 3, 4, 1]), dtype=flow.float32)

        >>> y = flow.eq(input, other)
        >>> y
        tensor([1, 1, 1, 0], dtype=oneflow.int8)

    """
    if type(input) == type(other):
        dtype_a = input.dtype
        dtype_b = other.dtype
        if dtype_a != dtype_b:
            _infer_dtype = _infered_binary_op_dtype(dtype_a, dtype_b)
            if dtype_a != _infer_dtype:
                input = input.to(_infer_dtype)
                other = other.to(_infer_dtype)
    return flow._C.equal(input, other)


@register_tensor_op("ne")
def ne_op(input, other):
    """
    Computes element-wise not equality.
    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.

    Args:
        input (oneflow.Tensor): the tensor to compare
        other (oneflow.Tensor, float or int): the target to compare

    Returns:

        - A boolean tensor that is True where :attr:`input` is not equal to :attr:`other` and False elsewhere

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.tensor(np.array([2, 3, 4, 5]), dtype=flow.float32)
        >>> other = flow.tensor(np.array([2, 3, 4, 1]), dtype=flow.float32)

        >>> y = flow.ne(input, other)
        >>> y
        tensor([0, 0, 0, 1], dtype=oneflow.int8)

    """
    if type(input) == type(other):
        dtype_a = input.dtype
        dtype_b = other.dtype
        if dtype_a != dtype_b:
            _infer_dtype = _infered_binary_op_dtype(dtype_a, dtype_b)
            if dtype_a != _infer_dtype:
                input = input.to(_infer_dtype)
                other = other.to(_infer_dtype)

    return flow._C.not_equal(input, other)


@register_tensor_op("lt")
def less_op(input, other):
    """Returns the truth value of :math:`input < other` element-wise.

    Args:
        input (oneflow.Tensor): A Tensor
        other (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: A Tensor with int8 type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.tensor(np.array([1, 2, 4]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.lt(input1, input2)
        >>> out
        tensor([0, 0, 1], dtype=oneflow.int8)

    """
    return flow._C.less(input, other)


@register_tensor_op("le")
def less_equal_op(input, other):
    """Returns the truth value of :math:`input <= other` element-wise.

    Args:
        input (oneflow.Tensor): A Tensor
        other (oneflow.Tensor): A Tensor

    Returns:
        oneflow.Tensor: A Tensor with int8 type.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input1 = flow.tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        >>> input2 = flow.tensor(np.array([1, 1, 4]).astype(np.float32), dtype=flow.float32)

        >>> out = flow.le(input1, input2)
        >>> out
        tensor([1, 0, 1], dtype=oneflow.int8)

    """
    return flow._C.less_equal(input, other)


@register_tensor_op("ne")
def ne_op(input, other):
    """
    Computes element-wise not equality.
    The second argument can be a number or a tensor whose shape is broadcastable with the first argument.

    Args:
        input (oneflow.Tensor): the tensor to compare
        other (oneflow.Tensor, float or int): the target to compare

    Returns:

        - A boolean tensor that is True where :attr:`input` is not equal to :attr:`other` and False elsewhere

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.tensor(np.array([2, 3, 4, 5]), dtype=flow.float32)
        >>> other = flow.tensor(np.array([2, 3, 4, 1]), dtype=flow.float32)

        >>> y = flow.ne(input, other)
        >>> y
        tensor([0, 0, 0, 1], dtype=oneflow.int8)

    """
    return flow._C.not_equal(input, other)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

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
from oneflow.nn.modules.utils import _single


def _input_args_is_int(args):
    return all((isinstance(x, int) for x in args))


def _input_args_is_tuple_int(args):
    return all((_input_args_is_int(x) for x in args))


def _input_args_is_flow_size(args):
    return all((isinstance(x, flow.Size) for x in args)) and len(args) == 1


@register_tensor_op("expand")
def expand_op(input, *sizes):
    """This operator expand the input tensor to a larger size.

    Passing -1 as the size for a dimension means not changing the size of that dimension.

    Tensor can be also expanded to a larger number of dimensions and the new ones will be appended at the front.

    For the new dimensions, the size cannot be set to -1.

    Args:
        input (oneflow.Tensor): The input Tensor.
        *sizes  (oneflow.Size or int): The desired expanded size.

    Returns:
        oneflow.Tensor: The result Tensor.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        >>> x = np.array([[[[0, 1]],
        ...               [[2, 3]],
        ...               [[4, 5]]]]).astype(np.int32)
        >>> input = flow.Tensor(x)
        >>> input.shape
        oneflow.Size([1, 3, 1, 2])
        >>> out = input.expand(1, 3, 2, 2)
        >>> out.shape
        oneflow.Size([1, 3, 2, 2])

    """
    if _input_args_is_int(sizes):
        expand_size = _single(sizes)
    elif _input_args_is_tuple_int(sizes):
        expand_size = _single(*sizes)
    elif _input_args_is_flow_size(sizes):
        expand_size = _single(*sizes)[0]
    else:
        raise ValueError("input sizes parameter is not illegal!")

    if input.dtype == flow.int8:
        input = flow.cast(input, flow.int32)
    return flow._C.expand(input, expand_size)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

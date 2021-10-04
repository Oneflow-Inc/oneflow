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
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import _single


def _input_args_is_int(args):
    return all((isinstance(x, int) for x in args))


def _input_args_is_tuple_int(args):
    return all((_input_args_is_int(x) for x in args))


def _input_args_is_flow_size(args):
    return all((isinstance(x, flow.Size) for x in args)) and len(args) == 1


@register_tensor_op("repeat")
def repeat_op(input, *sizes):
    """This operator repeat the input tensor to a larger size along the specified dimensions.

    Args:
        x (oneflow.Tensor): The input Tensor.
        *size (flow.Size or int): The number of times to repeat this tensor along each dimension

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
        >>> out = input.repeat(1, 1, 2, 2)
        >>> out.shape
        oneflow.Size([1, 3, 2, 4])
    """

    if input.ndim == 0 and input.numel() == 1:
        input = input.unsqueeze(0)

    if _input_args_is_int(sizes):
        sizes = _single(sizes)
    elif _input_args_is_tuple_int(sizes):
        sizes = _single(*sizes)
    elif _input_args_is_flow_size(sizes):
        sizes = _single(*sizes)[0]
    else:
        raise ValueError("input sizes parameter is not illegal!")

    for repeat_v in sizes:
        assert repeat_v > 0
    input_shape = input.shape
    assert len(sizes) >= len(input_shape)
    in_reshape = []
    out_reshape = []
    expand_dim = []
    diff = len(sizes) - len(input_shape)
    for i in range(len(sizes) - 1, -1, -1):
        if i >= diff:
            if sizes[i] > 1:
                if input_shape[i - diff] > 1:
                    in_reshape.insert(0, input_shape[i - diff])
                    in_reshape.insert(0, 1)
                    expand_dim.insert(0, input_shape[i - diff])
                    expand_dim.insert(0, sizes[i])
                    out_reshape.insert(0, input_shape[i - diff] * sizes[i])
                else:
                    in_reshape.insert(0, input_shape[i - diff])
                    expand_dim.insert(0, sizes[i])
                    out_reshape.insert(0, sizes[i])
            else:
                in_reshape.insert(0, input_shape[i - diff])
                expand_dim.insert(0, input_shape[i - diff])
                out_reshape.insert(0, input_shape[i - diff])
        else:
            expand_dim.insert(0, sizes[i])
            out_reshape.insert(0, sizes[i])
    new_tensor = flow.reshape(input, in_reshape)
    tmp_tensor = new_tensor.expand(*expand_dim)
    out = flow.reshape(tmp_tensor, out_reshape)
    return out


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

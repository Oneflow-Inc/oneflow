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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
import oneflow.python.framework.id_util as id_util
from typing import Optional, Sequence
from oneflow.python.nn.modules.utils import _check_axis


class ReduceSum(Module):
    def __init__(self, axis: int = None, keepdims: bool = False) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self._op = (
            flow.builtin_op("reduce_sum")
            .Input("input_tensor")
            .Output("output_tensor")
            .Attr("keepdims", keepdims)
            .Build()
        )

    def forward(self, input):
        axis = _check_axis(self.axis, input.shape)
        if len(axis) == 0:
            return input
        else:
            return self._op(input, axis=axis)[0]


@oneflow_export("reduce_sum")
@register_tensor_op("reduce_sum")
@experimental_api
def reduce_sum_op(input, axis=None, keepdims=False):
    r"""This operator computes the sum of elements across dimensions of a tensor

    Args:
        input (Tensor): A Tensor
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the sum value is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Tensor. Defaults to False.

    Returns:
        Tensor: The result of sum on the specified axis of input Tensor
    
    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np
        np_arr = np.random.randn(2,3,4,5)
        input = flow.Tensor(np_arr)
        output = flow.reduce_sum(input, 1, True)
        # equal to np.sum(np_arr, 1, keepdims=True)

    """
    return ReduceSum(axis, keepdims)(input)


@oneflow_export("reduce_mean")
@register_tensor_op("reduce_mean")
@experimental_api
def reduce_mean_op(input, axis=None, keepdims=False):
    r"""This operator computes the mean of input Tensor along the specified axis

    Args:
        input (Tensor): A Tensor
        axis (Optional[Union[collections.Sized, int]], optional): The dimension along which the mean value is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Tensor. Defaults to False.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Returns:
        Tensor: The result of average on the specified axis of input Tensor

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        np_arr = np.random.randn(2,3,4,5)
        input = flow.Tensor(np_arr)
        output = flow.reduce_mean(input, 1, True)
        # equal to np.mean(input_arr, 1, keepdims=True)

    """
    reduce_sum = ReduceSum(axis, keepdims)(input)
    axis = _check_axis(axis, input.shape)
    reduce_count = 1
    if len(axis) == 0:
        reduce_count = input.nelemenet()
    else:
        for i in axis:
            reduce_count *= input.shape[i]
    reduce_mean = reduce_sum / reduce_count
    return reduce_mean


class ReduceVariance(Module):
    def __init__(self, axis: int = None, keepdims: bool = False) -> None:
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, input):
        axis = _check_axis(self.axis, input.shape)
        if isinstance(axis, list) and len(axis) == 0:
            return flow.experimental.zeros(size=input.shape)
        else:
            return flow.experimental.sub(
                flow.experimental.reduce_mean(
                    flow.experimental.square(input), axis, self.keepdims
                ),
                flow.experimental.square(
                    flow.experimental.reduce_mean(input, axis, self.keepdims)
                ),
            )


@oneflow_export("reduce_variance")
@register_tensor_op("reduce_variance")
@experimental_api
def reduce_variance_op(input, axis=None, keepdims=False):
    r"""This operator computes the variance of input Tensor along the specified axis

    The equation is:

    .. math::

        out=\frac{1}{n}*\sum_{i=1}^{n}(x_i-mean)^2

    Args:
        input (Tensor): A Blob
        axis (Optional[Union[int, Sequence[int]]], optional): The dimension along which the variance is computed. Defaults to None.
        keepdims (bool, optional): Whether to keep the reduced dimension in the output Tensor. Defaults to False.

    Returns:
        Tensor: The result of variance on the specified axis of input Tensor

    For example:

    .. code-block:: python

        import oneflow as flow
        import numpy as np

        np_arr = np.random.randn(2,3,4,5)
        input = flow.Tensor(np_arr)
        output = flow.reduce_variance(input, 1, True)
        # equal to np.var(input_arr, 1, keepdims=True)

    """
    return ReduceVariance(axis, keepdims)(input)

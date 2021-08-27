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
from typing import Optional, Sequence

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import Tensor, register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Cat(Module):
    def __init__(self, dim=0) -> None:
        super().__init__()
        self.axis = dim

    def forward(self, inputs):
        if len(inputs) == 1:
            return inputs[0]
        axis = self.axis
        assert len(inputs) >= 2
        if axis < 0:
            axis += len(inputs[0].shape)
        assert axis >= 0 and axis < len(
            inputs[0].shape
        ), "axis must be in range [0, num_axes of inputs)"
        first_input_shape = inputs[0].shape
        dynamic_dim_size = 0
        for input in inputs:
            assert len(input.shape) == len(first_input_shape)
            for i in range(len(input.shape)):
                if i == axis:
                    dynamic_dim_size += input.shape[i]
                else:
                    assert input.shape[i] == first_input_shape[i]
        return flow.F.concat(inputs, axis=axis, max_dim_size=dynamic_dim_size)


def concat_op(inputs, dim=0):
    """Concatenate two or more `Tensor` s at specified axis.

    Analogous to `numpy.concatenate <https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html>`_

    Args:
        inputs: a `list` of `Tensor`
        dim: a `int`.

    Returns:
        A `Tensor`

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> flow.enable_eager_execution()
        >>> import numpy as np

        >>> input1 = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> input2 = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        >>> input3 = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)

        >>> out = flow.cat([input1, input2, input3], dim=1)
        >>> out.shape
        flow.Size([2, 18, 5, 3])

    """
    return Cat(dim=dim)(inputs)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

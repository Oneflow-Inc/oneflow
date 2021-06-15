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

import sys
import random

import oneflow as flow
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op


@oneflow_export("bernoulli")
@register_tensor_op("bernoulli")
@experimental_api
class Bernoulli(Module):
    r"""Draws binaray random numbers (0 / 1) from a Bernoulli distribution.

    Analogous to `torch.bernoulli <https://pytorch.org/docs/stable/generated/torch.bernoulli.html>`_

    The attr:`input` tensor should be a tensor containing probabilities to be used for drawing the binary random number. 
    Hence, all values in attr:`input` have to be in the range: \test{0} \leg \test{input}_{i} leg \test{1}.

    The \text{i}^{th} element of the output tensor will draw a value \test{1} according to the \text{i}^{th} probability value 
    given in `input`.

    .. math::
        \text{out}_{i} = \test{Bernoulli}(p = \text{input}_{i})

    The returned `out` tensor only has values 0 or 1 and is of the same shape as `input`.

    `out` can have integral `dtype`, but `input` must have floating point `dtype`.

    Args:
        input (Tensor): the input tensor of probability values for the Bernoulli distribution.


    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> arr1 = np.array([[0.25, 0.45, 0.3],[0.55, 0.32, 0.13],[0.75, 0.15, 0.1]]).astype(np.float32)
        >>> input1 = flow.Tensor(arr1)
        >>> out1 = flow.bernoulli(input1)
        >>> print(out1.shape)
        flow.Size([3, 3])

        >>> input2 = flow.ones((3,3))
        >>> out2 = flow.bernoulli(input2)
        >>> print(out2.shape)
        flow.Size([3, 3]) 
        >>> print(out2)
        tensor([[ 1.,  1.,  1.],
        [ 1.,  1.,  1.],
        [ 1.,  1.,  1.]])

        >>> input3 = flow.zeros((3,3))
        >>> out3 = flow.bernoulli(input3)
        >>> print(out3.shape)
        flow.Size([3, 3]) 
        >>> print(out3.numpy())
        [[ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]]

    """
    
    def __init__(self) -> None:
        super().__init__()
        seed = random.randint(-sys.maxsize, sys.maxsize)
        self._op = (
            flow.builtin_op("bernoulli")
            .Input("in")
            .Output("out")
            .Attr("seed", seed)
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=False)

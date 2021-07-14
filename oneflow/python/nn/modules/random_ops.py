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


@oneflow_export("bernoulli")
@experimental_api
def bernoulli(input, *, generator=None, out=None):
    r"""This operator returns a Tensor with binaray random numbers (0 / 1) from a Bernoulli distribution.

    Args:
        input(Tensor) - the input tensor of probability values for the Bernoulli distribution
        generator: (optional) – a pseudorandom number generator for sampling
        out (Tensor, optional) – the output tensor.

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow.experimental as flow

        >>> arr = np.array(
        ...    [
        ...        [1.0, 1.0, 1.0],
        ...        [1.0, 1.0, 1.0],
        ...        [1.0, 1.0, 1.0],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> y = flow.bernoulli(x)
        >>> y
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]], dtype=oneflow.float32)


    """
    return flow.F.bernoulli(input, flow.float32, generator)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

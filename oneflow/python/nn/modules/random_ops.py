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


class Bernoulli(Module):
    def __init__(self, dtype=flow.float32):
        super().__init__()

        seed = random.randint(-sys.maxsize, sys.maxsize)
        self._op = (
            flow.builtin_op("bernoulli")
            .Input("in")
            .Output("out")
            .Attr("dtype", dtype)
            .Attr("has_seed", True)
            .Attr("seed", seed)
            .Build()
        )

    def forward(self, x):
        return self._op(x)[0]


@oneflow_export("bernoulli")
@experimental_api
def bernoulli(input, *, generator=None, out=None):
    r"""This operator returns a Blob with binaray random numbers (0 / 1) from a Bernoulli distribution.

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
        >>> flow.enable_eager_execution()

        >>> arr = np.array(
        ...    [
        ...        [0.25, 0.45, 0.3],
        ...        [0.55, 0.32, 0.13],
        ...        [0.75, 0.15, 0.1],
        ...    ]
        ... )
        >>> x = flow.Tensor(arr)
        >>> y = flow.bernoulli(x)
        >>> y
        tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 1., 0.]], dtype=oneflow.float32)


    """
    return Bernoulli()(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

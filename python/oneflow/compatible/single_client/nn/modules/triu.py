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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.framework.tensor import register_tensor_op
from oneflow.compatible.single_client.nn.module import Module


class Triu(Module):
    def __init__(self, diagonal=0):
        super().__init__()
        self.diagonal = diagonal

    def forward(self, x):
        return flow.F.triu(x, self.diagonal)


@register_tensor_op("triu")
def triu_op(x, diagonal=0):
    """Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, 
    the other elements of the result tensor out are set to 0.
    
    Args:
        input (Tensor): the input tensor. 
        diagonal (int, optional): the diagonal to consider

    For example:

    .. code-block:: python

        >>> import oneflow.compatible.single_client.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> x = flow.Tensor(np.ones(shape=(3, 3)).astype(np.float32))
        >>> flow.triu(x)
        tensor([[1., 1., 1.],
                [0., 1., 1.],
                [0., 0., 1.]], dtype=oneflow.float32)

    """
    return Triu(diagonal)(x)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

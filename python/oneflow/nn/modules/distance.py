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
from oneflow.framework.tensor import Tensor
from oneflow.nn.module import Module

from typing import Optional


class CosineSimilarity(Module):
    r"""    
    Returns cosine similarity between :math:`x_1` and :math:`x_2`, computed along `dim`.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.CosineSimilarity.html#torch.nn.CosineSimilarity

    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Input2: :math:`(\ast_1, D, \ast_2)`, same number of dimensions as x1, matching x1 size at dimension `dim`,
              and broadcastable with x1 at other dimensions.
        - Output: :math:`(\ast_1, \ast_2)`

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> from oneflow import nn
        >>> input1 = flow.randn(100, 128)
        >>> input2 = flow.randn(100, 128)
        >>> cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        >>> output = cos(input1, input2)
    """

    def __init__(self, dim: Optional[int] = 1, eps: Optional[float] = 1e-08,) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return flow._C.cosine_similarity(x1, x2, self.dim, self.eps)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

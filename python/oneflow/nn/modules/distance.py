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
from oneflow.nn.modules.module import Module

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


class PairwiseDistance(Module):
    r"""Computes the pairwise distance between vectors :math:`v_1`, :math:`v_2` using the p-norm:

    .. math ::
        \left \| x \right \| _p = (\sum_{i=1}^n \left | x_i \right |^p )^{\frac{1}{p}}

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.PairwiseDistance.html.

    Args:
        p (real): the norm degree. Default: 2
        eps (float, optional): Small value to avoid division by zero. Default: 1e-6
        keepdim (bool, optional): Determines whether or not to keep the vector dimension. Default: False

    Shape:
        - Input1: :math:`(N, D)` or :math:`(D)`, where N = batch dimension and D = vector dimension
        - Input2: :math:`(N, D)` or :math:`(D)`, same shape as the input1
        - Output: :math:`(N)` or :math:`()` based on input dimension. If keepdim is True, then :math:`(N, 1)` or :math:`(1)` based on input dimension.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> pdist = flow.nn.PairwiseDistance(p=2)
        >>> x1 = flow.arange(12).reshape(3, 4)
        >>> x2 = flow.arange(12).reshape(3, 4)
        >>> pdist(x1, x2)
        tensor([2.0000e-06, 2.0000e-06, 2.0000e-06], dtype=oneflow.float32)
        >>> pdist(x1, x2).shape
        oneflow.Size([3])

    """

    def __init__(
        self,
        p: Optional[float] = 2.0,
        eps: Optional[float] = 1e-06,
        keepdim: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.p = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return flow._C.pairwise_distance(
            x1, x2, p=self.p, eps=self.eps, keepdim=self.keepdim
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

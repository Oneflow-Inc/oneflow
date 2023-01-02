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
import oneflow
from oneflow.framework.docstr.utils import add_docstr

add_docstr(
    oneflow._C.cosine_similarity,
    r"""
    cosine_similarity(x1: Tensor, x2: Tensor, dim: int=1, eps: float=1e-8) -> Tensor

    Returns cosine similarity between ``x1`` and ``x2``, computed along dim. ``x1`` and ``x2`` must be broadcastable
    to a common shape. ``dim`` refers to the dimension in this common shape. Dimension ``dim`` of the output is
    squeezed (see :func:`oneflow.squeeze`), resulting in the
    output tensor having 1 fewer dimension.

    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.functional.cosine_similarity.html
    
    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
    
    Args:
        x1 (Tensor): First input.
        x2 (Tensor): Second input.
        dim (int, optional): Dimension along which cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8

    For examples:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        >>> input1 = flow.randn(100, 128)
        >>> input2 = flow.randn(100, 128)
        >>> output = F.cosine_similarity(input1, input2)
    """,
)

add_docstr(
    oneflow._C.pairwise_distance,
    r"""
    pairwise_distance(x1: Tensor, x2: Tensor, dim: float=2.0, eps: float=1e-6, keepdim: bool=False) -> Tensor
    Computes the pairwise distance between vectors :math:`v_1`, :math:`v_2` using the p-norm:

    .. math ::
        \left \| x \right \| _p = (\sum_{i=1}^n \left | x_i \right |^p )^{\frac{1}{p}}

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/generated/torch.nn.PairwiseDistance.html.

    Args:
        x1 (Tensor): First input.
        x2 (Tensor): Second input.
        p (real): the norm degree. Default: 2
        eps (float, optional): Small value to avoid division by zero. Default: 1e-6
        keepdim (bool, optional): Determines whether or not to keep the vector dimension. Default: False

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> x1 = flow.arange(12).reshape(3, 4)
        >>> x2 = flow.arange(12).reshape(3, 4)
        >>> output = flow.nn.functional.pairwise_distance(x1, x2, p=2)
        >>> output
        tensor([2.0000e-06, 2.0000e-06, 2.0000e-06], dtype=oneflow.float32)
        >>> output.shape
        oneflow.Size([3])

    """,
)

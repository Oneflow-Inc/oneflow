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
from typing import Optional
import oneflow as flow
from oneflow.framework.tensor import Tensor


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    r"""A simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    See :class:`oneflow.nn.Embedding` for more details.

    Args:
        input (oneflow.LongTensor): Tensor containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                     i.e. it remains as a fixed "pad".
        max_norm (float, optional): If given, each embedding vector with norm larger than max_norm is renormalized to have 
                                    norm max_norm
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of 
                                                frequency of the words in the mini-batch. Default False

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F

        >>> # a batch of 2 samples of 4 indices each
        >>> input = flow.tensor([[1,2,4,5],[4,3,2,9]])
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = flow.rand(10, 3)
        >>> output = F.embedding(input, embedding_matrix)
        >>> output.shape
        oneflow.Size([2, 4, 3])
        >>> # example with padding_idx
        >>> input = flow.tensor([[0,2,0,5]])
        >>> output = F.embedding(input, embedding_matrix, padding_idx=0)
        >>> output.shape
        oneflow.Size([1, 4, 3])
    """

    assert sparse is False, "Not support sparse=True yet!"
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(
                0
            ), "Padding_idx must be within num_embeddings"
            padding_idx = weight.size(0) + padding_idx

    if max_norm is not None:
        with flow.no_grad():
            weight = flow._C.embedding_renorm_(weight, input, max_norm, norm_type)

    if padding_idx is None and not scale_grad_by_freq:
        return flow._C.gather(weight, input, axis=0)
    else:
        return flow._C.embedding(weight, input, padding_idx, scale_grad_by_freq)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

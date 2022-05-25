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
from typing import List, Optional, Tuple

import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn.module import Module


class Embedding(Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size.

    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.

    Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                    therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                    i.e. it remains as a fixed "pad". For a newly constructed Embedding,
                                    the embedding vector at :attr:`padding_idx` will default to all zeros,
                                    but can be updated to another value to be used as the padding vector.
    
    For example:

    .. code-block:: python
        
        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> indices = flow.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int)
        >>> m = flow.nn.Embedding(10, 3)
        >>> y = m(indices)

    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: Optional[float] = None,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        assert max_norm is None, "Not support max_norm yet!"
        assert norm_type is None, "Not support norm_type yet!"
        assert scale_grad_by_freq is False, "Not support scale_grad_by_freq=True yet!"
        assert sparse is False, "Not support sparse=True yet!"
        if _weight is None:
            self.weight = flow.nn.Parameter(Tensor(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = flow.nn.Parameter(_weight)
        self.sparse = sparse

    def reset_parameters(self) -> None:
        flow.nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with flow.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, indices):
        res = flow._C.gather(self.weight, indices, axis=0)
        return res


def embedding(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=None,
    scale_grad_by_freq=False,
    sparse=False,
):
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
    assert max_norm is None, "Not support max_norm yet!"
    assert norm_type is None, "Not support norm_type yet!"
    assert scale_grad_by_freq is False, "Not support scale_grad_by_freq=True yet!"
    assert sparse is False, "Not support sparse=True yet!"
    if padding_idx is not None:
        weight[padding_idx].fill_(0)
    res = flow._C.gather(weight, input, axis=0)
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

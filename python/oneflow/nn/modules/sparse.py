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
import os
from typing import List, Optional, Tuple

import oneflow as flow
from oneflow.framework.tensor import Tensor
from oneflow.nn.modules.module import Module


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
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm` is renormalized to have 
                                    norm :attr:`max_norm`
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default :attr:`2`.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of 
                                                frequency of the words in the mini-batch. Default :attr:`False`

    For example:

    .. code-block:: python
        
        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> indices = flow.tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int)
        >>> m = flow.nn.Embedding(10, 3)
        >>> y = m(indices)
        
    ..
        Feature Stage of Operator [Embedding].
        - Maintainer List [@EsdeathYZH]
        - Current Stage [ ]
        - Alpha Stage Check List [ ]
          - API(Compatible with PyTorch 1.11, anything incompatible must be noted in API Doc.)[Yes]
          - Doc(API Doc must be provided and showed normally on the web page.)[Yes]
          - Functionality and its' Test [ ]
            - Functionality is highly compatiable with PyTorch 1.11. [Yes]
            - eager local [Yes] [@EsdeathYZH]
              - forward [Yes]
              - backward [Yes]
              - gpu [Yes]
              - cpu [Yes]
            - graph local [ ] [@BBuf, @strint, @hjchen2]
              - forward [Yes]
              - backward [ ]
              - gpu [Yes]
              - cpu [Yes]
          - Exception Handling
            - Exception Message and Hint must be provided [ ]
        - Beta Stage Check List [ ]
          - API(High compatibility with PyTorch 1.11, shouldn't have anything incompatible for a naive reason.)[ ]
          - Doc(Same standard as Alpha Stage)[ ]
          - Functionality and its' Test [ ]
            - eager global [ ]
              - forward [ ]
              - backward [ ]
              - gpu [ ]
              - cpu [ ]
            - graph gloal [ ]
              - forward [ ]
              - backward [ ]
              - gpu [ ]
              - cpu [ ]
          - Performance and Scalability(Must be evaluated.)[ ]
            - CUDA kernel [ ]
            - CPU kernel [ ]
            - N nodes M devices [ ]
          - Exception Handling [ ]
            - Exception Message and Hint must be provided [ ]
            - Try you best to do Exception Recovery [ ]
        - Stable Stage Check List [ ]
          - API(Same standard as Beta Stage)[ ]
          - Doc(Same standard as Beta Stage)[ ]
          - Functionality and its' Test [ ]
            - fp16 and AMP [ ]
            - NHWC [ ]
          - Performance and Scalability(Must be evaluated.)[ ]
          - Exception Handling [ ]

    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
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
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        assert sparse is False, "Not support sparse=True yet!"
        if _weight is None:
            self.weight = flow.nn.Parameter(
                flow.empty((num_embeddings, embedding_dim), **factory_kwargs)
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = flow.nn.Parameter(_weight)
        self.sparse = sparse

    def reset_parameters(self) -> None:
        if os.getenv("ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT", "0") == "1":
            return
        flow.nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with flow.no_grad():
                self.weight[self.padding_idx] = 0

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)

    def forward(self, indices):
        if self.max_norm is not None:
            with flow.no_grad():
                flow._C.embedding_renorm_(
                    self.weight, indices, self.max_norm, self.norm_type
                )
        if self.padding_idx is None and not self.scale_grad_by_freq:
            return flow._C.gather(self.weight, indices, axis=0)
        else:
            return flow._C.embedding(
                self.weight, indices, self.padding_idx, self.scale_grad_by_freq
            )


def embedding(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
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

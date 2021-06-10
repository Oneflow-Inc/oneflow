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

from oneflow.python.framework.tensor import Tensor
from oneflow.python.oneflow_export import experimental_api, oneflow_export
from oneflow.python.nn.module import Module

from typing import Optional, List, Tuple


@oneflow_export("nn.Onehot")
@experimental_api
class Onehot(Module):
    """This operator generates a onehot Blob from input Blob.

    If input Blob's rank is `N`, the corresponding onehot Blob's rank is `N+1`. The new axis is generated on the specified dimension according to the parameter `axis`.

    The locations represented by `indices` take value `on_value`, while other locations take `off_value`

    Args:
        indices (oneflow._oneflow_internal.BlobDesc): The input Blob.
        depth (int): The length of onehot Blob.
        on_value (Union[int, float], optional): The fill value when `indices[i] == i`. Defaults to 1.
        off_value (Union[int, float], optional): The fill value when `indice[i] != i`. Defaults to 0.
        axis (int, optional): The specified dimension that the new axis is generated on. Defaults to -1.
        dtype (Optional[flow.dtype], optional): The output data type, it can be "oneflow.int32", "oneflow.int64", "oneflow.float", "oneflow.double". Defaults to None.
        name (Optional[str], optional): The name for the operation. Defaults to None.

    Note:

        The data type of input blob should be `int32` or `int64`

    For example:

    Example 1:

        >>> import oneflow.experimental as flow
        >>> import oneflow.experimental.nn as nn
        >>> import oneflow.typing as tp
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> x = np.array([0, 3, 1, 2]).astype(np.int32)
        >>> m = nn.Onehot(x, 5, -1, flow.int32)
        [[1 0 0 0 0]
        [0 0 0 1 0]
        [0 1 0 0 0]
        [0 0 1 0 0]]

    Example 2:

        >>> import oneflow.experimental as flow
        >>> import oneflow.experimental.nn as nn
        >>> import oneflow.typing as tp
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> x = np.array([0, 3, 1, 2]).astype(np.int32)
        >>> m = nn.Onehot(indices=x,
                                depth=5,
                                axis=0,
                                dtype=flow.int32)
       [[1 0 0 0]
       [0 0 1 0]
       [0 0 0 1]
       [0 1 0 0]
       [0 0 0 0]]

    """

    def __init__(
        self,
        indices: oneflow._oneflow_internal.BlobDesc,
        depth: int,
        on_value: Union[int, float] = 1,
        off_value: Union[int, float] = 0,
        axis: int = -1,
        dtype: Optional[flow.dtype] = None,
        name: Optional[str] = None,
    ) -> oneflow._oneflow_internal.BlobDesc:
        super().__init__()

    self._op = (
        flow.builtin_op("OneHot_")
        .Op("one_hot")
        .Input("indices", [indices])
        .Attr("depth", int(depth))
        .Attr("floating_on_value", float(on_value))
        .Attr("integer_on_value", int(on_value))
        .Attr("floating_off_value", float(off_value))
        .Attr("integer_off_value", int(off_value))
        .Attr("dtype", dtype)
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )

    def forward(self, x):
        res = self._op()
        if self.axis != (self.out_ndims - 1):
            dim_list = list(range(0, self.out_ndims))
            dim_list.insert(self.axis, self.out_ndims - 1)
            dim_list.pop()
            res = flow.transpose(res, self.dim_list)
        return res


@oneflow_export("nn.Embedding")
class Embedding(Module):
    r"""A simple lookup table that stores embeddings of a fixed dictionary and size.

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
        >>> import oneflow.experimental as flow
        >>> flow.enable_eager_execution()

        >>> indices = flow.Tensor([[1, 2, 4, 5], [4, 3, 2, 9]], dtype=flow.int)
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
        self._gather_op = (
            flow.builtin_op("gather")
            .Input("in")
            .Input("indices")
            .Output("out")
            .Attr("axis", int(0))
            .Build()
        )

    def reset_parameters(self) -> None:
        flow.nn.init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        # TODO padding_idx rely on tensor slice
        if self.padding_idx is not None:
            with flow.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, indices):
        res = self._gather_op(self.weight, indices)[0]
        return res


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

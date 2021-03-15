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
import oneflow.python.framework.id_util as id_util

from oneflow.python.framework.tensor import Tensor
from oneflow.python.oneflow_export import oneflow_export
from oneflow.python.nn.module import Module

from typing import Optional, List, Tuple


@oneflow_export("nn.Gather")
class Gather(Module):
    def __init__(
        self, axis: int = 0, sparse_grad: bool = False, name: Optional[str] = None,
    ):
        super().__init__()
        self._op = (
            flow.builtin_op("gather")
            .Name(name if name is not None else id_util.UniqueStr("Gather_"))
            .Input("in")
            .Input("indices")
            .Output("out")
            .Attr("axis", int(axis))
            .Build()
        )

    def forward(self, x, indices):
        res = self._op(x, indices)[0]
        return res


@oneflow_export("nn.Embedding")
class Embedding(Module):
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
        name: Optional[str] = None,
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
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = flow.nn.Parameter(Tensor(num_embeddings, embedding_dim))
            # TODO(Liangdepeng)
            # self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embedding_dim,
            ], "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = flow.nn.Parameter(_weight)
            # TODO(Liangdepeng)
            # self._fill_padding_idx_with_zero()
        self.sparse = sparse

        self.gather_op = Gather(
            name=name if name is not None else id_util.UniqueStr("Embedding_")
        )

    def forward(self, indices):
        res = self.gather_op(self.weight, indices)
        return res

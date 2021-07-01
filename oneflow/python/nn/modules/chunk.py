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
from oneflow.python.framework.tensor import Tensor
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module


class Chunk(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input, chunks, dim):
        if dim is not None:
            assert input.shape[dim] > 0, "chunk expects at least a 1-dimensional tensor"

            assert chunks > 0, "chunk expects `chunks` to be greater than 0"

            channel = input.dim()
            dim_size = input.shape[dim]
            chunk_size = (int)((dim_size + chunks - 1) / chunks)
            last_chunk_size = (
                chunk_size if dim_size % chunk_size == 0 else dim_size % chunk_size
            )
            chunk_dim_dict = {}
            tup_ndim = []
            splits = []

            for chunk in range(0, chunks):
                if dim_size % chunk_size == 0:
                    start = chunk * chunk_size
                    stop = (chunk + 1) * chunk_size
                    step = 1
                else:
                    start = (
                        chunk * chunk_size
                        if chunk < chunks - 1
                        else dim_size - 1 - last_chunk_size
                    )
                    stop = (
                        (chunk + 1) * chunk_size if chunk < chunks - 1 else dim_size - 1
                    )
                    step = 1
                chunk_dim_dict.setdefault(dim, []).append([start, stop, step])

            for k, v in chunk_dim_dict.items():
                for v_chunk in v:
                    tup_list = []
                    for i in range(0, channel):
                        if i != dim:
                            tup_list.append([None, None, None])
                        else:
                            tup_list.append(v_chunk)
                    splits.append(
                        flow.experimental.slice(input, slice_tup_list=tup_list)
                    )

            return splits


@oneflow_export("chunk")
@experimental_api
def chunk_op(input, chunks, dim):
    r"""Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor. Last chunk will be smaller if the tensor size along the given dimension dim is not divisible by chunks.

    Args:
        input (oneflow.experimental.Tensor): The tensor to split.
        chunks (int): Number of chunks to return.
        dim (int): Dimension along which to split the tensor.

    Returns:
        List of Tensors.

    For example:

    .. code-block:: python
    
        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        >>> input = flow.Tensor([[1, 2, 3], [7, 8, 9]], dtype = flow.float32)
        >>> out = flow.chunk(input, chunks=2, dim=0)
        >>> out
        [tensor([[1., 2., 3.]], dtype=oneflow.float32), tensor([[7., 8., 9.]], dtype=oneflow.float32)]
    """
    return Chunk()(input, chunks, dim)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=False)

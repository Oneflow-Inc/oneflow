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
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.nn.module import Module
from oneflow.python.framework.tensor import register_tensor_op
from oneflow.python.ops.array_ops import argwhere, gather, gather_nd


class MaskedSelect(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        assert len(x.shape) == len(
            mask.shape
        ), f"The dim of masked_select module's inputs can not match, please check!"
        broadcast_like_shape = []
        broadcast_x_axes = []
        broadcast_mask_axes = []
        for i in range(len(x.shape)):
            max_dim = max(x.shape[i], mask.shape[i])
            broadcast_like_shape.append(max_dim)
            if max_dim != x.shape[i]:
                broadcast_x_axes.append(i)
            if max_dim != mask.shape[i]:
                broadcast_mask_axes.append(i)
        broadcast_like_tensor = flow.experimental.zeros(
            tuple(broadcast_like_shape), dtype=flow.float32
        )
        broadcast_like_tensor = broadcast_like_tensor.to(x.device.type)
        broadcast_like_tensor.requires_grad = x.requires_grad or mask.requires_grad
        if len(broadcast_x_axes) != 0:
            broadcast_x = flow.experimental.broadcast_like(
                x, broadcast_like_tensor, broadcast_axes=tuple(broadcast_x_axes)
            )

        if len(broadcast_mask_axes) != 0:
            broadcast_mask = flow.experimental.broadcast_like(
                mask, broadcast_like_tensor, broadcast_axes=tuple(broadcast_mask_axes)
            )

        res = flow.F.mul(broadcast_x, broadcast_mask)
        indices = flow.experimental.argwhere(res)
        gather_res = flow.experimental.gather_nd(res, indices)
        gather_res = gather_res.flatten()
        return gather_res


@oneflow_export("masked_select")
@register_tensor_op("masked_select")
@experimental_api
def masked_select_op(x, mask):
    r"""

    Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor(In oneFlow BoolTensor is replaced by Int8Tensor).

    The shapes of the mask tensor and the input tensor don’t need to match, but they must be broadcastable.

    Args:
        input (Tensor): the input tensor.
        mask (Tensor): the tensor containing the binary mask to index with

    For example:

    .. code-block:: python

        >>> import oneflow.experimental as flow
        >>> import numpy as np
        >>> flow.enable_eager_execution()

        

    """
    return MaskedSelect()(x, mask)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

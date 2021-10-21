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
from typing import Optional, Sequence

import oneflow as flow
from oneflow.nn.module import Module


def _calc_broadcast_axes(x, like_tensor):
    num_prepend = len(like_tensor.shape) - len(x.shape)
    prepend_shape = [1] * num_prepend + list(x.shape)
    broadcast_axes = [x for x in range(num_prepend)]
    for i in range(num_prepend, len(prepend_shape)):
        if prepend_shape[i] != like_tensor.shape[i]:
            if prepend_shape[i] != 1:
                raise RuntimeError(
                    f"output with shape {x.shape} doesn't match the broadcast shape {like_tensor.shape}"
                )
            else:
                broadcast_axes.append(i)
    return tuple(broadcast_axes)


class BroadCastLike(Module):
    def __init__(self, broadcast_axes: Optional[Sequence] = None) -> None:
        super().__init__()
        self.broadcast_axes = broadcast_axes

    def forward(self, x, like_tensor):
        if self.broadcast_axes is None:
            broadcast_axes = _calc_broadcast_axes(x, like_tensor)
        else:
            broadcast_axes = self.broadcast_axes
        return flow._C.broadcast_like(x, like_tensor, broadcast_axes=broadcast_axes)


def broadcast_like_op(x, like_tensor, broadcast_axes: Optional[Sequence] = None):
    """This operator broadcast tensor `x` to `like_tensor` according to the broadcast_axes. 

    Args:
        x (Tensor): The input Tensor. 
        like_tensor (Tensor): The like Tensor. 
        broadcast_axes (Optional[Sequence], optional): The axes you want to broadcast. Defaults to None.

    Returns:
        [Tensor]: Broadcasted input Tensor. 

    For example: 

    .. code:: python

        >>> import oneflow as flow 

        >>> x = flow.randn(3, 1, 1)
        >>> like_tensor = flow.randn(3, 4, 5)
        >>> broadcast_tensor = flow.broadcast_like(x, like_tensor, broadcast_axes=[1, 2]) 
        >>> broadcast_tensor.shape
        oneflow.Size([3, 4, 5])

    """
    return BroadCastLike(broadcast_axes=broadcast_axes)(x, like_tensor)

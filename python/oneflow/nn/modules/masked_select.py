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
from oneflow.framework.tensor import register_tensor_op


def masked_select_op(input, mask):
    assert len(input.shape) == len(
        mask.shape
    ), f"The dim of masked_select module's inputs can not match, please check!"
    broadcast_like_shape = []
    broadcast_x_axes = []
    broadcast_mask_axes = []
    for i in range(len(input.shape)):
        max_dim = max(input.shape[i], mask.shape[i])
        broadcast_like_shape.append(max_dim)
        if max_dim != input.shape[i]:
            broadcast_x_axes.append(i)
        if max_dim != mask.shape[i]:
            broadcast_mask_axes.append(i)
    broadcast_like_tensor = flow.zeros(
        tuple(broadcast_like_shape), dtype=flow.float32, device=input.device
    )
    broadcast_like_tensor.requires_grad = input.requires_grad or mask.requires_grad
    if len(broadcast_x_axes) != 0:
        input = flow.broadcast_like(
            input, broadcast_like_tensor, broadcast_axes=tuple(broadcast_x_axes)
        )
    if len(broadcast_mask_axes) != 0:
        mask = flow.broadcast_like(
            mask, broadcast_like_tensor, broadcast_axes=tuple(broadcast_mask_axes)
        )
    mask = mask.to(dtype=input.dtype)
    res = flow._C.mul(input, mask)
    indices = flow.argwhere(res)
    gather_res = flow._C.gather_nd(res, indices)
    return gather_res.flatten()


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

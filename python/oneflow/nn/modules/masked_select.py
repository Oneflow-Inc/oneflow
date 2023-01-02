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


def masked_select_op(input, mask):
    """

    Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor(In oneFlow BoolTensor is replaced by Int8Tensor).

    The shapes of the mask tensor and the input tensor don’t need to match, but they must be broadcastable.

    Args:
        input (Tensor): the input tensor.
        mask (Tensor): the tensor containing the binary mask to index with

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.tensor(np.array([[-0.4620, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]), dtype=flow.float32)
        >>> mask = input.gt(0.05)
        >>> out = flow.masked_select(input, mask)
        >>> out
        tensor([0.3139, 0.3898], dtype=oneflow.float32)
    """

    assert input.is_global == mask.is_global, (
        f"input tensor is %s tensor, but mask is %s tensor"
        % (
            "global" if input.is_global else "local",
            "global" if mask.is_global else "local",
        )
    )
    broadcast_shape = []
    input_shape_len = len(input.shape)
    mask_shape_len = len(mask.shape)
    input_shape = [input.shape[i] for i in range(input_shape_len)]
    input_shape.reverse()
    mask_shape = [mask.shape[i] for i in range(mask_shape_len)]
    mask_shape.reverse()
    for i in range(max(input_shape_len, mask_shape_len)):
        if i < input_shape_len and i < mask_shape_len:
            broadcast_shape.append(max(input_shape[i], mask_shape[i]))
        elif i < input_shape_len:
            broadcast_shape.append(input_shape[i])
        else:
            broadcast_shape.append(mask_shape[i])
    broadcast_shape.reverse()
    broadcast_input = input.expand(broadcast_shape)
    broadcast_mask = mask.expand(broadcast_shape)

    indices = flow.argwhere(broadcast_mask)
    gather_res = flow._C.gather_nd(broadcast_input, indices)

    return gather_res.flatten()


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

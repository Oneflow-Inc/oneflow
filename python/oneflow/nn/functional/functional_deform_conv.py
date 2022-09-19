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
from typing import Optional, Tuple, Union

import oneflow as flow
from oneflow.framework.tensor import Tensor


def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:

    use_mask = mask is not None

    if mask is None:
        mask = flow.zeros((input.shape[0], 0), dtype=input.dtype).to(input.device)

    stride_h = stride[0]
    stride_w = stride[1]
    pad_h = padding[0]
    pad_w = padding[1]
    dil_h = dilation[0]
    dil_w = dilation[1]
    weights_h, weights_w = weight.shape[-2:]

    # TODO(yzm):Support rectangle convolution
    # NOTE:Delete the following error reporting program after support rectangle convolution
    if weights_h != weights_w:
        raise RuntimeError(
            "Rectangle convolution is not currently supported, please try square convolution"
        )
    if bias is not None:
        if len(bias.shape) != 1 or bias.shape[0] != weight.shape[0]:
            raise RuntimeError("invalid bias shape:got:" f"{bias.shape}")

    if use_mask and len(mask.shape) != 4:
        raise RuntimeError("The dimension of mask tensor weight must be 4")
    if len(input.shape) != 4:
        raise RuntimeError("The dimension of input tensor weight must be 4")
    if len(weight.shape) != 4:
        raise RuntimeError("The dimension of weight tensor weight must be 4")
    if len(offset.shape) != 4:
        raise RuntimeError("The dimension of offset tensor weight must be 4")

    _, n_in_channels, _, _ = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "The shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            f"Got offset.shape[1]={offset.shape[1]}, while 2 * weight.size[2] * weight.size[3]={2 * weights_h * weights_w}"
        )

    return flow._C.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    )

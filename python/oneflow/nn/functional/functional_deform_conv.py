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
    r"""
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.

    Args:
        input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]):
            offsets to be applied for each position in the convolution kernel.
        weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]): convolution weights,
            split into groups of size (in_channels // groups)
        bias (Tensor[out_channels]): optional bias of shape (out_channels,). Default: None
        stride (int or Tuple[int, int]): distance between convolution centers. Default: 1
        padding (int or Tuple[int, int]): height/width of padding of zeroes around
            each image. Default: 0
        dilation (int or Tuple[int, int]): the spacing between kernel elements. Default: 1
        mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]):
            masks to be applied for each position in the convolution kernel. Default: None

    Returns:
        Tensor[batch_sz, out_channels, out_h, out_w]: result of convolution

    Examples::
        >>> import oneflow as flow
        >>> import oneflow.nn.functional as F
        >>> input = flow.rand(4, 3, 10, 10)
        >>> kh, kw = 3, 3
        >>> weight = flow.rand(5, 3, kh, kw)
        >>> # offset and mask should have the same spatial size as the output
        >>> # of the convolution. In this case, for an input of 10, stride of 1
        >>> # and kernel size of 3, without padding, the output size is 8
        >>> offset = flow.rand(4, 2 * kh * kw, 8, 8)
        >>> mask = flow.rand(4, kh * kw, 8, 8)
        >>> out = F.deform_conv2d(input, offset, weight, mask=mask)
        >>> out.size()
        oneflow.Size([4, 5, 8, 8])
    """
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

    # TODO(yzm): Support rectangle convolution
    if weights_h != weights_w:
        raise NotImplementedError("Rectangle convolution is not supported currently.")

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

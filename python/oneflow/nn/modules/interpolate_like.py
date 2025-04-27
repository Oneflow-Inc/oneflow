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
import math
import warnings
from typing import Optional, Tuple, Union

import oneflow as flow
from oneflow.framework.tensor import register_tensor_op
from oneflow.nn.modules.module import Module


class InterpolateLike:
    def __init__(
        self, mode: str = "nearest", align_corners: Optional[bool] = None,
    ):
        if mode in ("nearest", "area") and align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the interpolating modes: linear | bilinear | bicubic | trilinear"
            )
        self.mode = mode
        if align_corners == None:
            align_corners = False
        self.align_corners = align_corners
        if self.mode not in (
            "nearest",
            "bilinear",
            "linear",
            "area",
            "bicubic",
            "trilinear",
        ):
            raise ValueError(
                'interpolation must be "nearest" or "bilinear" or "linear" or "area" or "bicubic" or "trilinear".'
            )
        if self.mode == "nearest" and self.align_corners:
            raise ValueError('interpolation "nearest" does not support align_corners.')

    def forward(self, x, like):
        if len(x.shape) == 3 and self.mode == "bilinear":
            raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
        if len(x.shape) == 3 and self.mode == "trilinear":
            raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
        if len(x.shape) == 4 and self.mode == "linear":
            raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
        if len(x.shape) == 4 and self.mode == "trilinear":
            raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
        if len(x.shape) == 5 and self.mode == "linear":
            raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
        if len(x.shape) == 5 and self.mode == "bilinear":
            raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")

        dim = len(x.shape) - 2
        if len(x.shape) == 3 and self.mode == "nearest":
            return flow._C.upsample_nearest_1d(x, like, data_format="channels_first",)
        if len(x.shape) == 4 and self.mode == "nearest":
            return flow._C.upsample_nearest_2d(x, like, data_format="channels_first",)
        if len(x.shape) == 5 and self.mode == "nearest":
            return flow._C.upsample_nearest_3d(x, like, data_format="channels_first",)

        raise NotImplementedError(
            "Input Error: Only 3D, 4D and 5D input Tensors supported"
            " (got {}D) for the modes: nearest"
            " (got {})".format(len(x.shape), self.mode)
        )


def interpolate_like(
    input, like, mode="nearest", align_corners=None,
):
    """The interface is consistent with PyTorch.    
    
    The documentation is referenced from: https://pytorch.org/docs/1.10/_modules/torch/nn/functional.html#interpolate.
    

    Down/up samples the input to :Tensor:`like` shape.

    The algorithm used for interpolation is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric sampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for resizing are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

    Args:
        input (Tensor): the input tensor
        like (Tensor): the like tensor
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``

    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.

    For example:

    .. code-block:: python

        >>> import oneflow as flow
        >>> import numpy as np
        
        >>> input = flow.tensor(np.arange(1, 5).reshape((1, 1, 4)), dtype=flow.float32)
        >>> like = flow.randn(1, 1, 8)
        >>> output = flow.nn.functional.interpolate_like(input, like, mode="linear")
        >>> output
        tensor([[[1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.2500, 3.7500, 4.0000]]],
               dtype=oneflow.float32)

    """
    return InterpolateLike(mode=mode, align_corners=align_corners,).forward(input, like)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

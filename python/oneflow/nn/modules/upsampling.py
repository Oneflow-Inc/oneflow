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
from oneflow.nn.modules.module import Module


class Upsample(Module):
    """    
    Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/_modules/torch/nn/modules/upsampling.html.

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``

    Shape:
        - Input: :math:`(N, C, W_{in})`, :math:`(N, C, H_{in}, W_{in})` or :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, W_{out})`, :math:`(N, C, H_{out}, W_{out})`
          or :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

    .. math::
        D_{out} = \\left\\lfloor D_{in} \\times \\text{scale_factor} \\right\\rfloor

    .. math::
        H_{out} = \\left\\lfloor H_{in} \\times \\text{scale_factor} \\right\\rfloor

    .. math::
        W_{out} = \\left\\lfloor W_{in} \\times \\text{scale_factor} \\right\\rfloor

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, `bicubic`, and `trilinear`) don't proportionally
        align the output and input pixels, and thus the output values can depend
        on the input size. This was the default behavior for these modes up to
        version 0.3.1. Since then, the default behavior is
        ``align_corners = False``. See below for concrete examples on how this
        affects the outputs.

    .. note::
        If you want downsampling/general resizing, you should use :func:`~nn.functional.interpolate`.


    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow

        >>> input = flow.tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        >>> input = input.to("cuda")
        >>> m = flow.nn.Upsample(scale_factor=2.0, mode="nearest")
        >>> output = m(input)
        >>> output #doctest: +ELLIPSIS
        tensor([[[[1., 1., 2., 2.],
                  ...
                  [3., 3., 4., 4.]]]], device='cuda:0', dtype=oneflow.float32)

    """

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, ...]]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return flow.nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

    def extra_repr(self) -> str:
        if self.scale_factor is not None:
            info = "scale_factor=" + str(self.scale_factor)
        else:
            info = "size=" + str(self.size)
        info += ", mode=" + self.mode
        return info


class UpsamplingNearest2d(Upsample):
    """Applies a 2D nearest neighbor upsampling to an input signal composed of several input
    channels.

    To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
    as it's constructor argument.

    When :attr:`size` is given, it is the output size of the image `(h, w)`.

    Args:
        size (int or Tuple[int, int], optional): output spatial sizes
        scale_factor (float or Tuple[float, float], optional): multiplier for
            spatial size.

    .. warning::
        This class is deprecated in favor of :func:`~nn.functional.interpolate`.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

    .. math::
          H_{out} = \\left\\lfloor H_{in} \\times \\text{scale_factor} \\right\\rfloor

    .. math::
          W_{out} = \\left\\lfloor W_{in} \\times \\text{scale_factor} \\right\\rfloor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input = flow.tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        >>> input = input.to("cuda")
        >>> m = flow.nn.UpsamplingNearest2d(scale_factor=2.0)
        >>> output = m(input)
        >>> output #doctest: +ELLIPSIS
        tensor([[[[1., 1., 2., 2.],
                  ...
                  [3., 3., 4., 4.]]]], device='cuda:0', dtype=oneflow.float32)

    """

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[Tuple[float, float]] = None,
    ) -> None:
        super(UpsamplingNearest2d, self).__init__(size, scale_factor, mode="nearest")


class UpsamplingBilinear2d(Upsample):
    """Applies a 2D bilinear upsampling to an input signal composed of several input
    channels.

    To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
    as it's constructor argument.

    When :attr:`size` is given, it is the output size of the image `(h, w)`.

    Args:
        size (int or Tuple[int, int], optional): output spatial sizes
        scale_factor (float or Tuple[float, float], optional): multiplier for
            spatial size.

    .. warning::
        This class is deprecated in favor of :func:`~nn.functional.interpolate`. It is
        equivalent to ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where

    .. math::
        H_{out} = \\left\\lfloor H_{in} \\times \\text{scale_factor} \\right\\rfloor

    .. math::
        W_{out} = \\left\\lfloor W_{in} \\times \\text{scale_factor} \\right\\rfloor

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        
        >>> input = flow.tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        >>> input = input.to("cuda")
        >>> m = flow.nn.UpsamplingBilinear2d(scale_factor=2.0)
        >>> output = m(input)
        >>> output #doctest: +ELLIPSIS
        tensor([[[[1.0000, 1.3333, 1.6667, 2.0000],
                  ...
                  [3.0000, 3.3333, 3.6667, 4.0000]]]], device='cuda:0',
               dtype=oneflow.float32)

    """

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[Tuple[float, float]] = None,
    ) -> None:
        super(UpsamplingBilinear2d, self).__init__(
            size, scale_factor, mode="bilinear", align_corners=True
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

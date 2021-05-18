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
from oneflow.python.nn.module import Module
from oneflow.python.oneflow_export import oneflow_export, experimental_api
from oneflow.python.framework.tensor import register_tensor_op
from typing import Optional, Union, Tuple


@oneflow_export("nn.Upsample")
@experimental_api
class Upsample(Module):
    r"""Upsamples a given multi-channel 2D (spatial) data.

    The input data is assumed to be of the form
    `minibatch x channels x height x width`.
    Hence, for spatial inputs, we expect a 4D Tensor.

    The algorithms available for upsampling are nearest neighbor,
    bilinear, 4D input Tensor, respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int, int] optional):
            output spatial sizes
        scale_factor (float or Tuple[float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'bilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is ``'bilinear'``. 
            Default: ``False``

    Shape:
        - Input: : :math:`(N, C, H_{in}, W_{in})` 
        - Output: :math:`(N, C, H_{out}, W_{out})` , where

    .. math::
        D_{out} = \left\lfloor D_{in} \times \text{scale_factor} \right\rfloor

    .. math::
        H_{out} = \left\lfloor H_{in} \times \text{scale_factor} \right\rfloor

    .. math::
        W_{out} = \left\lfloor W_{in} \times \text{scale_factor} \right\rfloor

    .. note::
        If you want downsampling/general resizing, you should use :func:`~nn.functional.interpolate`.
    
    For example:

    .. code-block:: python

        import oneflow.experimental as flow
        
        input = flow.Tensor(np.arange(1, 5).reshape((1, 1, 2, 2)), dtype=flow.float32)
        input = input.to("cuda")
        m = flow.nn.Upsample(scale_factor=2.0, mode="nearest")
        output = m(input)

        # output.numpy()
        [[[[1.0, 1.0, 2.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [3.0, 3.0, 4.0, 4.0],
        [3.0, 3.0, 4.0, 4.0],]]]
    
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
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None

        self.mode = mode
        if align_corners == None:
            align_corners = False

        self.align_corners = align_corners
        self.height_scale = None
        self.width_scale = None

        if isinstance(self.scale_factor, float):
            self.height_scale = self.scale_factor
            self.width_scale = self.scale_factor
        elif isinstance(self.scale_factor, tuple):
            self.height_scale = self.scale_factor[0]
            self.width_scale = self.scale_factor[1]
        else:
            pass

        if self.mode != "nearest" and self.mode != "bilinear":
            raise ValueError('interpolation must be "nearest" or "bilinear".')

        if self.mode == "nearest" and self.align_corners:
            raise ValueError('interpolation "nearest" does not support align_corners.')

        self._op = (
            flow.builtin_op("upsample")
            .Input("x")
            .Output("y")
            .Attr("height_scale", float(1.0))
            .Attr("width_scale", float(1.0))
            .Attr("align_corners", self.align_corners)
            .Attr("data_format", "channels_first")
            .Attr("interpolation", self.mode)
            .Build()
        )

    def forward(self, x):
        assert (
            self.size != None or self.scale_factor != None
        ), f"size and scale_factor can not be none at the same time!"
        h, w = x.shape[2], x.shape[3]
        if self.height_scale == None:
            if isinstance(self.size, int):
                self.height_scale = 1.0 * self.size / h
            else:
                self.height_scale = 1.0 * self.size[0] / h
        if self.width_scale == None:
            if isinstance(self.size, int):
                self.width_scale = 1.0 * self.size / w
            else:
                self.width_scale = 1.0 * self.size[1] / w

        res = self._op(x, height_scale=self.height_scale, width_scale=self.width_scale)[
            0
        ]
        return res

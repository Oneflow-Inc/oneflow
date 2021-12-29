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
import os

import oneflow as flow
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from oneflow.nn.module import Module
from oneflow.nn.modules.utils import (
    _generate_output_size,
    _getint,
    _pair,
    _single,
    _triple,
)


class MaxPool1d(Module):
    r"""The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html#torch.nn.MaxPool1d

    Applies a 1D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`
    and output :math:`(N, C, L_{out})` can be precisely described as:

    .. math::
        out(N_i, C_j, k) = \max_{m=0, \ldots, \text{kernel\_size} - 1}
                input(N_i, C_j, stride \times k + m)

    If :attr:`padding` is non-zero, then the input is implicitly padded with minimum value on both sides
    for :attr:`padding` number of points. :attr:`dilation` is the stride between the elements within the
    sliding window. This `link`_ has a nice visualization of the pooling parameters.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    Args:
        kernel_size: The size of the sliding window, must be > 0.
        stride: The stride of the sliding window, must be > 0. Default value is :attr:`kernel_size`.
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        return_indices: If ``True``, will return the argmax along with the max values.
                        Useful for :class:`torch.nn.MaxUnpool1d` later
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This
                   ensures that every element in the input tensor is covered by a sliding window.

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, L_{out})`, where

          .. math::
              L_{out} = \left\lfloor \frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                    \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    For example: 

    .. code-block:: python 

        import oneflow as flow 
        import numpy as np

        of_maxpool1d = flow.nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        x = flow.Tensor(np.random.randn(1, 4, 4))
        y = of_maxpool1d(x)
        y.shape 
        oneflow.Size([1, 4, 4])

    """

    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride) if stride is not None else self.kernel_size
        data_format = "NCL"  # only support "NCL" for now !
        self.channel_pos = "channels_first" if data_format == "NCL" else "channels_last"
        self.dilation = _single(dilation)
        self.padding = _single(padding)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x):
        y, indice = flow._C.max_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=True,
            ceil_mode=self.ceil_mode,
            data_format=self.channel_pos,
        )
        if self.return_indices:
            return y, indice
        else:
            return y

    def extra_repr(self) -> str:
        return "kernel_size={}, stride={}, padding={}".format(
            self.kernel_size, self.stride, self.padding
        )


def get_dhw_offset(channel_pos):
    if channel_pos == "channels_first":
        return 2
    else:
        return 1


def get_ndim_pads_list(padding, dhw_offset, ndims):
    pads_list = []
    for i in range(len(padding)):
        pad = padding[i]
        if isinstance(pad, int):
            pad = [pad, pad]
        elif isinstance(pad, (list, tuple)):
            assert len(pad) == 2
            pad = [pad[0], pad[1]]
        else:
            raise ValueError("padding must be list tuple or int")
        if i in range(dhw_offset, dhw_offset + ndims):
            pads_list.append(pad)
        else:
            assert pad == [0, 0]
    return pads_list


def calc_pool_padding(padding, dhw_offset, ndims):
    if isinstance(padding, str):
        padding = "SAME_LOWER" if padding.upper() == "SAME" else padding
        assert padding.upper() in ["VALID", "SAME_LOWER", "SAME_UPPER"]
        padding_type = padding.lower()
        ndim_pads_list = [[0, 0]] * ndims
    elif isinstance(padding, (list, tuple)):
        padding_type = "customized"
        ndim_pads_list = get_ndim_pads_list(padding, dhw_offset, ndims)
    else:
        raise ValueError("padding must be str or a list.")
    return (padding_type, ndim_pads_list)


class MaxPool2d(Module):
    r"""The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

    Applies a 2D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            out(N_i, C_j, h, w) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                    & \text{input}(N_i, C_j, \text{stride[0]} \times h + m,
                                                   \text{stride[1]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly minimum value padded on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.
    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:
        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit minimum value padding to be added on both sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
                    \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
                    \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    For example:

    .. code-block:: python

        import oneflow as flow 
        import numpy as np

        of_maxpool2d = flow.nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        x = flow.Tensor(np.random.randn(1, 4, 4, 4))
        y = of_maxpool2d(x)
        y.shape 
        oneflow.Size([1, 4, 4, 4])

    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if (stride is not None) else _pair(kernel_size)
        self.ceil_mode = ceil_mode

        if os.getenv("ONEFLOW_ENABLE_NHWC") == "1":
            self.data_format = "NHWC"
            self.channel_pos = "channels_last"
            padding = _pair(padding)
            if len(padding) == 2:
                if self.data_format == "NCHW":
                    padding = (0, 0, padding[0], padding[1])
                elif self.data_format == "NHWC":
                    padding = (0, padding[0], padding[1], 0)
                else:
                    raise ValueError("error padding param!")
            self.padding = padding
            self.padding_type, pads_list = calc_pool_padding(
                padding, get_dhw_offset(self.channel_pos), 2
            )
            self.padding_before = [pad[0] for pad in pads_list]
            self.padding_after = [pad[1] for pad in pads_list]
            if return_indices == True:
                raise ValueError(
                    "MaxPool2d with NHWC data format don't support return indices for now."
                )
            if dilation != 1:
                raise ValueError(
                    "MaxPool2d with NHWC data format only support dilation == 1 for now."
                )
            self.dilation = _pair(dilation)

        else:
            self.data_format = "NCHW"
            self.channel_pos = "channels_first"
            self.padding = _pair(padding)
            self.dilation = _pair(dilation)
            self.return_indices = return_indices

    def forward(self, x):
        if self.data_format == "NCHW":
            y, indice = flow._C.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                return_indices=True,
                ceil_mode=self.ceil_mode,
                data_format=self.channel_pos,
            )
            if self.return_indices:
                return y, indice
            else:
                return y
        else:
            return flow._C.max_pool2d_nhwc(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding_type,
                padding_before=self.padding_before,
                padding_after=self.padding_after,
                data_format=self.channel_pos,
                ceil_mode=self.ceil_mode,
            )

    def extra_repr(self) -> str:
        return "kernel_size={}, stride={}, padding={}, dilation={}".format(
            self.kernel_size, self.stride, self.padding, self.dilation
        )


class MaxPool3d(Module):
    r"""The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html#torch.nn.MaxPool3d

    Applies a 3D max pooling over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:

    .. math::
        \begin{aligned}
            \text{out}(N_i, C_j, d, h, w) ={} & \max_{k=0, \ldots, kD-1} \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                              & \text{input}(N_i, C_j, \text{stride[0]} \times d + k,
                                                             \text{stride[1]} \times h + m, \text{stride[2]} \times w + n)
        \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly minimum value on both sides
    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit minimum value padding to be added on all three sides
        dilation: a parameter that controls the stride of elements in the window
        return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool3d` later
        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where

          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times
                (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1] \times
                (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2] \times
                (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    For example:

    .. code-block:: python

        import oneflow as flow 
        import numpy as np 

        of_maxpool3d = flow.nn.MaxPool3d(kernel_size=3, padding=1, stride=1)
        x = flow.Tensor(np.random.randn(1, 4, 4, 4, 4))
        y = of_maxpool3d(x)
        y.shape 
        oneflow.Size([1, 4, 4, 4, 4])

    """

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride) if (stride is not None) else _triple(kernel_size)
        data_format = "NCDHW"
        self.channel_pos = (
            "channels_last" if data_format == "NDHWC" else "channels_first"
        )
        self.dilation = _triple(dilation)
        self.padding = _triple(padding)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, x):
        y, indice = flow._C.max_pool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=True,
            ceil_mode=self.ceil_mode,
            data_format=self.channel_pos,
        )

        if self.return_indices:
            return y, indice
        else:
            return y

    def extra_repr(self) -> str:
        return "kernel_size={}, stride={}, padding={}, dilation={}".format(
            self.kernel_size, self.stride, self.padding, self.dilation
        )


class AvgPool1d(Module):
    r"""Applies a 1D average pooling over an input signal composed of several input planes.
    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and `kernel_size` :math:`k`
    can be precisely described as:
    
    .. math::
        out(N_i, C_j, l)  = \\frac{1}{k} \\sum_{m=0}^{k-1}
                               input(N_i, C_j, stride[0] \\times h + m, stride*l + m)
    
    If padding is non-zero, then the input is implicitly zero-padded on both sides for padding number of points.
    The parameters kernel_size, stride, padding can each be an int or a one-element tuple.
    
    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding or the
        input. Sliding windows that would start in the right padded region are ignored.
    
    Args:
        kernel_size: the size of the window.
        strides: the stride of the window. Default value is kernel_size.
        padding: implicit zero padding to be added on both sides.
        ceil_mode: when True, will use ceil instead of floor to compute the output shape.
        count_include_pad: when True, will include the zero-padding in the averaging calculation.
    
    For example: 

    .. code-block:: python 
        
        import oneflow as flow 
        import numpy as np

        m = flow.nn.AvgPool1d(kernel_size=3, padding=1, stride=1)
        x = flow.tensor(np.random.randn(1, 4, 4))
        y = m(x)
        y.shape 
        oneflow.Size([1, 4, 4])

    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        super().__init__()
        self.kernel_size = _single(kernel_size)
        data_format = "NCHW"  # only support "NCHW" for now !
        self.channel_pos = (
            "channels_first" if data_format == "NCHW" else "channels_last"
        )
        self.stride = _single(stride) if (stride is not None) else _single(kernel_size)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.padding = _single(padding)

    def forward(self, x):
        return flow._C.avg_pool1d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=0,
            data_format=self.channel_pos,
        )

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class AvgPool2d(Module):
    r"""Performs the 2d-average pooling on the input.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_{out}, W_{out})` and `kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \times h + m, stride[1] \times w + n)

    Args:
        kernel_size (Union[int, Tuple[int, int]]):  An int or list of ints that has length 1, 2. The size of the window for each dimension of the input Tensor.
        strides (Union[int, Tuple[int, int]]): An int or list of ints that has length 1, 2. The stride of the sliding window for each dimension of the input Tensor.
        padding (Tuple[int, int]): An int or list of ints that has length 1, 2. Implicit zero padding to be added on both sides.
        ceil_mode (bool, default to False): When True, will use ceil instead of floor to compute the output shape.

    For example:

    .. code-block:: python

        import oneflow as flow 
        import numpy as np 

        m = flow.nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        x = flow.tensor(np.random.randn(1, 4, 4, 4))
        y = m(x)   
        y.shape
        oneflow.Size([1, 4, 4, 4])

    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int = 0,
    ):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if (stride is not None) else _pair(kernel_size)
        self.ceil_mode = ceil_mode

        if os.getenv("ONEFLOW_ENABLE_NHWC") == "1":
            self.data_format = "NHWC"
            self.channel_pos = "channels_last"
            assert isinstance(padding, int) or isinstance(
                padding, tuple
            ), "padding can only int int or tuple of 2 ints."
            padding = _pair(padding)
            if len(padding) == 2:
                if self.data_format == "NCHW":
                    padding = (0, 0, padding[0], padding[1])
                elif self.data_format == "NHWC":
                    padding = (0, padding[0], padding[1], 0)
                else:
                    raise ValueError("error padding param!")
            self.padding = padding

            if not count_include_pad:
                raise ValueError(
                    "AvgPool2d with NHWC data format don't support count_include_pad for now."
                )
            if divisor_override != 0:
                raise ValueError(
                    "AvgPool2d with NHWC data format don't support divisor_override for now."
                )

            # TODO(yaochi): align with pytorch when padding is asymmetric
            self._padding_type, _pads_list = calc_pool_padding(
                padding, get_dhw_offset(self.channel_pos), 2
            )
            self._padding_before = [pad[0] for pad in _pads_list]
            self._padding_after = [pad[1] for pad in _pads_list]
        else:
            self.data_format = "NCHW"
            self.channel_pos = "channels_first"
            self.padding = _pair(padding)
            self.count_include_pad = count_include_pad
            self.divisor_override = int(divisor_override)

    def forward(self, x):
        if self.data_format == "NCHW":
            return flow._C.avg_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                divisor_override=self.divisor_override,
                data_format=self.channel_pos,
            )
        else:
            return flow._C.avg_pool2d_nhwc(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self._padding_type,
                padding_before=self._padding_before,
                padding_after=self._padding_after,
                ceil_mode=self.ceil_mode,
                data_format=self.channel_pos,
            )

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class AvgPool3d(Module):
    r"""Applies a 3D average pooling over an input signal composed of several input planes.
    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,
    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and `kernel_size` :math:`(kD, kH, kW)`
    can be precisely described as:
    
    .. math::
        out(N_i, C_j, d, h, w)  = \\frac{1}{kD * kH * kW } \\sum_{k=0}^{kD-1} \\sum_{m=0}^{kH-1} \\sum_{n=0}^{kW-1}
                               input(N_i, C_j, stride[0] \\times d + k, stride[1] \\times h + m, stride[2] \\times w + n)
    
    If padding is non-zero, then the input is implicitly zero-padded on all three sides for padding number of points.
    
    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding or the
        input. Sliding windows that would start in the right padded region are ignored.
    
    Args:
        kernel_size: the size of the window.
        strides:  the stride of the window. Default value is kernel_size.
        padding:  implicit zero padding to be added on all three sides.
        ceil_mode:  when True, will use ceil instead of floor to compute the output shape.
        count_include_pad: when True, will include the zero-padding in the averaging calculation.
        divisor_override: if specified, it will be used as divisor, otherwise kernel_size will be used.
    
    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`

        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})`, where
        
          .. math::
              D_{out} = \\left\\lfloor\\frac{D_{in} + 2 \\times \\text{padding}[0] - \\text{kernel_size}[0]}{\\text{stride}[0]} + 1\\right\\rfloor
        
          .. math::
              H_{out} = \\left\\lfloor\\frac{H_{in} + 2 \\times \\text{padding}[1] - \\text{kernel_size}[1]}{\\text{stride}[1]} + 1\\right\\rfloor
        
          .. math::
              W_{out} = \\left\\lfloor\\frac{W_{in} + 2 \\times \\text{padding}[2] - \\text{kernel_size}[2]}{\\text{stride}[2]} + 1\\right\\rfloor
    
    For example:
    
    .. code-block:: python
    
        import oneflow as flow
        import numpy as np
        
        m = flow.nn.AvgPool3d(kernel_size=(2,2,2),padding=(0,0,0),stride=(1,1,1))
        x = flow.tensor(np.random.randn(9, 7, 11, 32, 20))
        y = m(x)
        y.shape
        oneflow.Size([9, 7, 10, 31, 19])

    """

    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int = 0,
    ):
        super().__init__()
        self.kernel_size = _triple(kernel_size)
        data_format = "NCHW"  # only support "NCHW" for now !
        self.channel_pos = (
            "channels_first" if data_format == "NCHW" else "channels_last"
        )
        self.stride = _triple(stride) if (stride is not None) else _triple(kernel_size)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = int(divisor_override)
        self.padding = _triple(padding)

    def forward(self, x):
        return flow._C.avg_pool3d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
            data_format=self.channel_pos,
        )

    def extra_repr(self) -> str:
        return (
            "kernel_size={kernel_size}, stride={stride}, padding={padding}"
            ", ceil_mode={ceil_mode}".format(**self.__dict__)
        )


class AdaptiveAvgPool1d(Module):
    """Applies a 1D adaptive average pooling over an input signal composed of several input planes.

    The output size is H, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size H

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> m = nn.AdaptiveAvgPool1d(5)
        >>> input = flow.Tensor(np.random.randn(1, 64, 8))
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 5])

    """

    def __init__(self, output_size: _size_1_t) -> None:
        super().__init__()
        assert output_size is not None, "'output_size' cannot be NoneType"
        self.output_size = _single(output_size)

    def forward(self, x):
        assert (
            len(x.shape) == 3 and len(self.output_size) == 1
        ), "the length of 'output_size' does not match the input size, 1 expected"
        assert isinstance(
            self.output_size[0], int
        ), "numbers in 'output_size' should be integer"
        return flow._C.adaptive_avg_pool1d(x, output_size=self.output_size)


def adaptive_avg_pool1d(input, output_size):
    """Applies a 1D adaptive average pooling over an input signal composed of several input planes.

    See :mod:`oneflow.nn.AdaptiveAvgPool1d`

    Args:
        input: input tensor
        output_size: the target output size (single integer)
    """
    return AdaptiveAvgPool1d(output_size)(input)


class AdaptiveAvgPool2d(Module):
    """Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> m = nn.AdaptiveAvgPool2d((5,7))
        >>> input = flow.Tensor(np.random.randn(1, 64, 8, 9))
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 5, 7])

        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = flow.Tensor(np.random.randn(1, 64, 10, 9))
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 7, 7])

        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = flow.Tensor(np.random.randn(1, 64, 10, 9))
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 10, 7])

    """

    def __init__(self, output_size) -> None:
        super().__init__()
        assert output_size is not None, "'output_size' cannot be NoneType"
        self.output_size = _pair(output_size)

    def forward(self, x):
        assert (
            len(x.shape) == 4
        ), f"expected 4-dimensional tensor, but got {len(x.shape)}-dimensional tensor"
        new_output_size = _generate_output_size(x.shape, self.output_size)
        return flow._C.adaptive_avg_pool2d(x, output_size=new_output_size)


def adaptive_avg_pool2d(input, output_size):
    """Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    See :mod:`oneflow.nn.AdaptiveAvgPool2d`

    Args:
        input: input tensor
        output_size: the target output size (single integer or double-integer tuple)
    """
    return AdaptiveAvgPool2d(output_size)(input)


class AdaptiveAvgPool3d(Module):
    """Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    The output is of size D x H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the form D x H x W.
                     Can be a tuple (D, H, W) or a single number D for a cube D x D x D.
                     D, H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    For example:

    .. code-block:: python

        >>> import numpy as np
        >>> import oneflow as flow
        >>> import oneflow.nn as nn

        >>> m = nn.AdaptiveAvgPool3d((5,7,9))
        >>> input = flow.Tensor(np.random.randn(1, 64, 8, 9, 10))
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 5, 7, 9])

        >>> m = nn.AdaptiveAvgPool3d(7)
        >>> input = flow.Tensor(np.random.randn(1, 64, 10, 9, 8))
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 7, 7, 7])

        >>> m = nn.AdaptiveAvgPool3d((7, None, None))
        >>> input = flow.Tensor(np.random.randn(1, 64, 10, 9, 8))
        >>> output = m(input)
        >>> output.size()
        oneflow.Size([1, 64, 7, 9, 8])

    """

    def __init__(self, output_size) -> None:
        super().__init__()
        assert output_size is not None, "'output_size' cannot be NoneType"
        self.output_size = _triple(output_size)

    def forward(self, x):
        assert (
            len(x.shape) == 5
        ), f"expected 5-dimensional tensor, but got {len(x.shape)}-dimensional tensor"
        new_output_size = _generate_output_size(x.shape, self.output_size)
        return flow._C.adaptive_avg_pool3d(x, output_size=new_output_size)


def adaptive_avg_pool3d(input, output_size):
    """Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    See :mod:`oneflow.nn.AdaptiveAvgPool3d`

    Args:
        input: input tensor
        output_size: the target output size (single integer or triple-integer tuple)
    """
    return AdaptiveAvgPool3d(output_size)(input)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

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
import oneflow

# oneflow._C.max_poolXd returns a TensorTuple, to align torch,
# here we return different result according to the param `return_indices`.
def max_pool1d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False,
    data_format="channels_first",
):
    r"""
    max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False,ceil_mode=False, data_format="channels_first")

    Applies a 1D max pooling over an input signal composed of several input
    planes.

    The documentation is referenced from: https://pytorch.org/docs/master/generated/torch.nn.functional.max_pool1d.html.

    .. note::
        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from
        what seen in :class:`~oneflow.nn.MaxPool1d`, and will change in a future release.

    See :class:`~oneflow.nn.MaxPool1d` for details.

    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in_channels} , iW)`, minibatch dim optional.
        kernel_size: the size of the window. Can be a single number or a tuple `(kW,)`
        stride: the stride of the window. Can be a single number or a tuple `(sW,)`. Default: :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        return_indices: If ``True``, will return the argmax along with the max values.Useful for :class:`oneflow.nn.functional.max_unpool1d` later.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.
    """
    _max_pool_out = oneflow._C.max_pool1d(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        ceil_mode,
        data_format,
    )
    if return_indices:
        return _max_pool_out
    else:
        return _max_pool_out[0]


def max_pool2d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False,
    data_format="channels_first",
):
    r"""
    max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False,data_format="channels_first")

    Applies a 2D max pooling over an input signal composed of several input
    planes.

    The documentation is referenced from: https://pytorch.org/docs/master/generated/torch.nn.functional.max_pool2d.html.

    .. note::
        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from
        what seen in :class:`~oneflow.nn.MaxPool2d`, and will change in a future release.

    See :class:`~oneflow.nn.MaxPool2d` for details.

    Args:
        input: input tensor :math:`(\text{minibatch} , \text{in_channels} , iH , iW)`, minibatch dim optional.
        kernel_size: size of the pooling region. Can be a single number or a tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        return_indices: If ``True``, will return the argmax along with the max values.Useful for :class:`oneflow.nn.functional.max_unpool2d` later.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.
    """
    _max_pool_out = oneflow._C.max_pool2d(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        ceil_mode,
        data_format,
    )
    if return_indices:
        return _max_pool_out
    else:
        return _max_pool_out[0]


def max_pool3d(
    x,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    return_indices=False,
    ceil_mode=False,
    data_format="channels_first",
):
    r"""
    max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, data_format="channels_first")

    Applies a 3D max pooling over an input signal composed of several input
    planes.

    The documentation is referenced from: https://pytorch.org/docs/master/generated/torch.nn.functional.max_pool3d.html.

    .. note::
        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from
        what seen in :class:`~oneflow.nn.MaxPool3d`, and will change in a future release.

    See :class:`~oneflow.nn.MaxPool3d` for details.

    Args:
        input: input tensor :math:`(\text{minibatch} , \text{in_channels} , iD, iH , iW)`, minibatch dim optional.
        kernel_size: size of the pooling region. Can be a single number or a tuple `(kT, kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`
        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.
        dilation: The stride between elements within a sliding window, must be > 0.
        return_indices: If ``True``, will return the argmax along with the max values.Useful for :class:`~oneflow.nn.functional.max_unpool3d` later.
        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This ensures that every element in the input tensor is covered by a sliding window.
    """
    _max_pool_out = oneflow._C.max_pool3d(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        ceil_mode,
        data_format,
    )
    if return_indices:
        return _max_pool_out
    else:
        return _max_pool_out[0]


def adaptive_max_pool1d(input, output_size, return_indices: bool = False):
    r"""Applies a 1D adaptive max pooling over an input signal composed of
    several input planes.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.functional.adaptive_max_pool1d.html

    See :class:`~oneflow.nn.AdaptiveMaxPool1d` for details and output shape.

    Args:
        output_size: the target output size (single integer)
        return_indices: whether to return pooling indices. Default: ``False``

    """

    _out = oneflow._C.adaptive_max_pool1d(input, output_size)
    if return_indices:
        return _out
    else:
        return _out[0]


def adaptive_max_pool2d(input, output_size, return_indices: bool = False):
    r"""Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.functional.adaptive_max_pool2d.html

    See :class:`~oneflow.nn.AdaptiveMaxPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``

    """
    _out = oneflow._C.adaptive_max_pool2d(input, output_size)
    if return_indices:
        return _out
    else:
        return _out[0]


def adaptive_max_pool3d(input, output_size, return_indices: bool = False):
    r"""Applies a 3D adaptive max pooling over an input signal composed of
    several input planes.

    The documentation is referenced from:
    https://pytorch.org/docs/1.10/generated/torch.nn.functional.adaptive_max_pool3d.html

    See :class:`~oneflow.nn.AdaptiveMaxPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``

    """

    _out = oneflow._C.adaptive_max_pool3d(input, output_size)
    if return_indices:
        return _out
    else:
        return _out[0]

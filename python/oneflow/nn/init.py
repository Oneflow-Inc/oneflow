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
import os

import oneflow as flow
from oneflow.ops.initializer_util import CalcGain


def calculate_gain(nonlinearity, param=None):
    return CalcGain(nonlinearity, param)


def uniform_(tensor, a=0.0, b=1.0):
    with flow.no_grad():
        return tensor.uniform_(a, b)


def normal_(tensor, mean=0.0, std=1.0):
    with flow.no_grad():
        return tensor.normal_(mean, std)


def xavier_uniform_(tensor, gain=1.0, *, data_format="NCHW"):
    r"""
    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/nn.init.html.

    Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `flow.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = flow.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    with flow.no_grad():
        return tensor.xavier_uniform_(gain, data_format=data_format)


def xavier_normal_(tensor, gain=1.0, *, data_format="NCHW"):
    r"""
    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/nn.init.html.

    Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan_in} + \text{fan_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `flow.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = flow.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    with flow.no_grad():
        return tensor.xavier_normal_(gain, data_format=data_format)


def kaiming_uniform_(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", *, data_format="NCHW"
):
    r"""
    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/nn.init.html.

    Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `flow.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = flow.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    with flow.no_grad():
        return tensor.kaiming_uniform_(a, mode, nonlinearity, data_format=data_format)


def kaiming_normal_(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", *, data_format="NCHW"
):
    r"""
    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/stable/nn.init.html.
    
    Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `flow.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = flow.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    if os.getenv("ONEFLOW_ENABLE_NHWC") == "1":
        data_format = "NHWC"
    with flow.no_grad():
        return tensor.kaiming_normal_(a, mode, nonlinearity, data_format=data_format)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with flow.no_grad():
        return tensor.trunc_normal_(mean, std, a, b)


def constant_(tensor, val):
    with flow.no_grad():
        return tensor.fill_(val)


def ones_(tensor):
    with flow.no_grad():
        return tensor.fill_(1)


def zeros_(tensor):
    with flow.no_grad():
        return tensor.fill_(0)


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.ndimension()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )
    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.ndimension() > 2:
        for s in tensor.size()[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return (fan_in, fan_out)

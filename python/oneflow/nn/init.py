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
from oneflow.framework.tensor import _copy_from_numpy_to_eager_local_tensor, Tensor
from oneflow.ops.util.initializer_util import calc_gain as calculate_gain
import oneflow.ops.initializer_register as initializer_register


def _init_by_initializer_conf(tensor, initializer_conf, random_seed=None):
    # NOTE: initializing weight should not enable autograd mode
    if random_seed is None:
        random_seed = flow.default_generator.initial_seed()
    shape = tuple(tensor.shape)
    initializer = initializer_register.get_initializer(
        initializer_conf, random_seed, shape
    )

    np_arr = initializer_register.generate_values_by_initializer(
        initializer, shape, tensor.dtype
    )
    with flow.no_grad():
        if tensor.is_global:
            src_tensor = flow.tensor(np_arr)
            src_tensor = src_tensor.to_global(
                placement=tensor.placement,
                sbp=tuple(flow.sbp.broadcast for _ in range(len(tensor.sbp))),
            )
            tensor.copy_(src_tensor)
        else:
            _copy_from_numpy_to_eager_local_tensor(
                tensor, np_arr,
            )
    return tensor


def uniform_(tensor, a=0.0, b=1.0):
    if isinstance(a, Tensor):
        assert a.ndim == 0 and a.nelement() == 1, "a must be a number or scalar tensor!"
        a = a.numpy().item()
    if isinstance(b, Tensor):
        assert b.ndim == 0 and b.nelement() == 1, "b must be a number or scalar tensor!"
        b = b.numpy().item()
    initializer_conf = initializer_register.random_uniform_initializer(
        minval=a, maxval=b, dtype=tensor.dtype
    )
    return _init_by_initializer_conf(tensor, initializer_conf)


def normal_(tensor, mean=0.0, std=1.0):
    initializer_conf = initializer_register.random_normal_initializer(mean, std)
    return _init_by_initializer_conf(tensor, initializer_conf)


def xavier_uniform_(tensor, gain=1.0, *, data_format="NCHW"):
    r"""
    Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/nn.init.html.

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `flow.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = flow.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    initializer_conf = initializer_register.xavier_initializer(
        tensor.shape, gain=gain, data_format=data_format, distribution="random_uniform"
    )
    return _init_by_initializer_conf(tensor, initializer_conf)


def xavier_normal_(tensor, gain=1.0, *, data_format="NCHW"):
    r"""
    Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan_in} + \text{fan_out}}}

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/nn.init.html.

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `flow.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = flow.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    initializer_conf = initializer_register.xavier_initializer(
        tensor.shape, gain=gain, data_format=data_format, distribution="random_normal"
    )
    return _init_by_initializer_conf(tensor, initializer_conf)


def orthogonal_(tensor, gain=1.0):
    r"""
    Fills the input `Tensor` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/nn.init.html.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> w = flow.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    with flow.no_grad():
        return tensor.orthogonal_(gain)


def kaiming_uniform_(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", *, data_format="NCHW"
):
    r"""
    Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan_mode}}}
    
    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/nn.init.html.

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
    if os.getenv("ONEFLOW_ENABLE_NHWC") == "1":
        data_format = "NHWC"
    initializer_conf = initializer_register.kaiming_initializer(
        tensor.shape,
        a=a,
        mode=mode,
        nonlinearity=nonlinearity,
        data_format=data_format,
        distribution="random_uniform",
    )
    return _init_by_initializer_conf(tensor, initializer_conf)


def kaiming_normal_(
    tensor, a=0, mode="fan_in", nonlinearity="leaky_relu", *, data_format="NCHW"
):
    r"""    
    Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan_mode}}}

    The interface is consistent with PyTorch.
    The documentation is referenced from: https://pytorch.org/docs/1.10/nn.init.html.

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
    initializer_conf = initializer_register.kaiming_initializer(
        tensor.shape,
        a=a,
        mode=mode,
        nonlinearity=nonlinearity,
        data_format=data_format,
        distribution="random_normal",
    )
    return _init_by_initializer_conf(tensor, initializer_conf)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    initializer_conf = initializer_register.truncated_normal_initializer(
        mean=mean, std=std, a=a, b=b,
    )
    return _init_by_initializer_conf(tensor, initializer_conf)


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

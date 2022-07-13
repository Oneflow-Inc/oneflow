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

import functools
import math
from typing import Optional, Sequence, Union

import numpy as np

import oneflow as flow
from oneflow.ops.util.initializer_util import (
    get_random_distribution,
    get_data_format,
    calc_fan,
    calc_gain,
)
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.framework.dtype as dtype_util


_init_map = {}


def register_initializer(flow_initializer):
    def deco(func):
        _init_map[flow_initializer] = func
        return func

    return deco


def constant_initializer(
    value: float = 0, dtype: flow.dtype = flow.float
) -> initializer_conf_util.InitializerConf:
    """Initializer that generates blob with constant values.

    Args:
        value (float, optional): A Python scalar. All elements of the initialized variable . Defaults to 0.
        dtype (flow.dtype, optional): Default data type. Defaults to flow.float.

    Raises:
        NotImplementedError:  Do not support such data type.

    Returns:
        initializer_conf_util.InitializerConf:  An InitializerConf object.
    """
    initializer = initializer_conf_util.InitializerConf()
    if dtype in [flow.float, flow.double]:
        setattr(initializer.constant_conf, "value", float(value))
    elif dtype in [flow.int8, flow.int32, flow.int64]:
        setattr(initializer.constant_int_conf, "value", int(value))
    else:
        raise NotImplementedError("Do not support such data type")
    return initializer


def zeros_initializer(
    dtype: flow.dtype = flow.float,
) -> initializer_conf_util.InitializerConf:
    """Initializer that generates blobs initialized to 0

    Args:
        dtype (flow.dtype, optional): Default data type. Defaults to flow.float.

    Returns:
        initializer_conf_util.InitializerConf: constant_initializer
    """
    return constant_initializer(0.0, dtype)


def ones_initializer(
    dtype: flow.dtype = flow.float,
) -> initializer_conf_util.InitializerConf:
    """Initializer that generates blobs initialized to 1.

    Args:
        dtype (flow.dtype, optional): Default data type. Defaults to flow.float.

    Returns:
        initializer_conf_util.InitializerConf: constant_initializer
    """
    return constant_initializer(1.0, dtype)


def random_uniform_initializer(
    minval: float = 0, maxval: float = 1, dtype: flow.dtype = flow.float
) -> initializer_conf_util.InitializerConf:
    """Initializer that generates blobs with a uniform distribution. 

    Args:
        minval (float, optional): A python scalar. Lower bound of the range of random values to generate. Defaults to 0.
        maxval (float, optional): A python scalar. Upper bound of the range of random values to generate. Defaults to 1.
        dtype (flow.dtype, optional): Default data type. Defaults to flow.float.

    Raises:
        NotImplementedError: Do not support such data type.

    Returns:
        initializer_conf_util.InitializerConf:  Initial configuration
    """
    assert minval <= maxval
    initializer = initializer_conf_util.InitializerConf()
    if dtype in [flow.float, flow.double]:
        setattr(initializer.random_uniform_conf, "min", float(minval))
        setattr(initializer.random_uniform_conf, "max", float(maxval))
    elif dtype in [flow.int8, flow.int32, flow.int64]:
        setattr(initializer.random_uniform_int_conf, "min", int(minval))
        setattr(initializer.random_uniform_int_conf, "max", int(maxval))
    else:
        raise NotImplementedError("Do not support such data type")
    return initializer


def random_normal_initializer(
    mean: float = 0.0, stddev: float = 1.0,
) -> initializer_conf_util.InitializerConf:
    """Initializer that generates blob with a normal distribution.

    Args:
        mean (float, optional): A python scalar. Mean of the random values to generate.. Defaults to 0.0.
        stddev (float, optional): A python scalar. Standard deviation of the random values to generate. Defaults to 1.0.
        seed (Optional[int], optional): None. Not support yet. Defaults to None.
        dtype (Optional[flow.dtype], optional): . Defaults to None.

    Returns:
        initializer_conf_util.InitializerConf: Initial configuration
    """
    initializer = initializer_conf_util.InitializerConf()
    setattr(initializer.random_normal_conf, "mean", float(mean))
    setattr(initializer.random_normal_conf, "std", float(stddev))
    return initializer


def xavier_initializer(
    shape: Sequence[int],
    gain: float = 1.0,
    data_format: str = "NCHW",
    distribution: str = "random_normal",
):
    r"""
    Initializer weight according to the method described in `Understanding the
    difficulty of training deep feedforward neural networks - Glorot, X. & Bengio,
    Y. (2010)`, using a normal or uniform distribution.

    Also known as Glorot initialization.

    When distribution is "random_normal", the resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math:: 

        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    When distribution is "random_uniform", the resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math:: 

        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Args:
        shape (Sequence[int]): Blob shape.
        gain (float, optional): an optional scaling factor. default: 1.0
        data_format (str, optional):  'NCHW', 'NHWC'. Defaults to "NCHW".
        distribution (str, optional): 'random_normal' or 'random_uniform'. Defaults to "random_normal".

    Returns:
        initializer_conf_util.InitializerConf:  Initial configuration
    """
    assert isinstance(shape, (tuple, flow.Size))
    elem_cnt = functools.reduce(lambda a, b: a * b, shape, 1)
    assert elem_cnt > 0
    assert distribution in ["random_normal", "random_uniform"]
    fan = calc_fan(shape, "fan_sum", get_data_format(data_format))
    std = gain * math.sqrt(2.0 / fan)
    if distribution == "random_normal":
        return random_normal_initializer(0.0, std)
    elif distribution == "random_uniform":
        bound = math.sqrt(3.0) * std
        return random_uniform_initializer(-bound, bound)
    else:
        raise NotImplementedError(
            "xavier_initializer only support `random_norm` or `random_uniform`"
        )


def kaiming_initializer(
    shape: Sequence[int],
    a: float = 0.0,
    mode: str = "fan_in",
    nonlinearity="leaky_relu",
    data_format: str = "NCHW",
    distribution: str = "random_normal",
) -> None:
    r"""Initialize weight according to the method described in `Delving deep into
    rectifiers: Surpassing human-level performance on ImageNet classification`
    - He, K. et al. (2015), using a normal or uniform distribution.

    When distribution is "random_normal", the resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math:: 

        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    When distribution is "random_uniform", the resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math:: 

        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    If mode is "fan_in", the "n" is the number of input units in the weight Blob. 

    If mode is "fan_out", the "n" is the number of output units in the weight Blob. 

    Args:
        shape (Sequence[int]): Blob shape.
        a (float, optional): the negative slope of the rectifier used after this layer
            (only used with ``'leaky_relu'``)
        mode (str, optional): 'fan_in', 'fan_out'. Defaults to "fan_in".
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
        data_format (str, optional):  'NCHW', 'NHWC'. Defaults to "NCHW".
        distribution (str, optional): 'random_normal' or 'random_uniform'. Defaults to "random_normal".

    Returns:
        initializer_conf_util.InitializerConf:  Initial configuration
    """
    assert isinstance(shape, (tuple, flow.Size))
    elem_cnt = functools.reduce(lambda a, b: a * b, shape, 1)
    assert elem_cnt > 0, "cannot initializing zero-element tensor"
    assert distribution in ["random_normal", "random_uniform"]
    assert mode in ["fan_in", "fan_out"]
    assert data_format in ["NCHW", "NHWC"]
    fan = calc_fan(shape, mode, get_data_format(data_format))
    gain = calc_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    if distribution == "random_normal":
        return random_normal_initializer(0.0, std)
    elif distribution == "random_uniform":
        bound = math.sqrt(3.0) * std
        return random_uniform_initializer(-bound, bound)
    else:
        raise NotImplementedError("Only support normal and uniform distribution")


def truncated_normal_initializer(
    mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0,
):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean (float, optional): the mean of the normal distribution
        std (float, optional): the standard deviation of the normal distribution
        a (float, optional): the minimum cutoff value
        b (float, optional): the maximum cutoff value
    """
    initializer = initializer_conf_util.InitializerConf()
    trunc_normal_conf = getattr(initializer, "trunc_normal_conf")
    # set norm_conf
    norm_conf = getattr(trunc_normal_conf, "norm_conf")
    setattr(norm_conf, "mean", float(mean))
    setattr(norm_conf, "std", float(std))
    # set max/min
    setattr(trunc_normal_conf, "min", float(a))
    setattr(trunc_normal_conf, "max", float(b))
    return initializer


@register_initializer("constant_conf")
@register_initializer("constant_int_conf")
def ConstantInitializerImpl(
    initializer_conf: Union[
        initializer_conf_util.ConstantInitializerConf,
        initializer_conf_util.ConstantIntInitializerConf,
    ],
    random_seed: int,
    var_blob_shape: Sequence[int],
):
    return lambda length: np.full((length,), initializer_conf.value)


@register_initializer("random_normal_conf")
def RandomNormalInitializerImpl(
    initializer_conf: initializer_conf_util.RandomNormalInitializerConf,
    random_seed: int,
    var_blob_shape: Sequence[int],
):
    rng = np.random.default_rng(random_seed)
    return lambda length: rng.normal(
        loc=initializer_conf.mean, scale=initializer_conf.std, size=length
    )


@register_initializer("random_uniform_conf")
def RandomUniformInitializerImpl(
    initializer_conf: initializer_conf_util.RandomUniformIntInitializerConf,
    random_seed: int,
    var_blob_shape: Sequence[int],
):
    rng = np.random.default_rng(random_seed)
    return lambda length: rng.uniform(
        low=initializer_conf.min,
        high=np.nextafter(initializer_conf.max, float("inf")),
        size=length,
    )


@register_initializer("random_uniform_int_conf")
def RandomUniformIntInitializerImpl(
    initializer_conf: initializer_conf_util.RandomUniformIntInitializerConf,
    random_seed: int,
    var_blob_shape: Sequence[int],
):
    rng = np.random.default_rng(random_seed)
    return lambda length: rng.integers(
        low=initializer_conf.min, high=initializer_conf.max, size=length
    )


@register_initializer("trunc_normal_conf")
def TruncNormalInitializerImpl(
    initializer_conf: initializer_conf_util.TruncNormalInitializerConf,
    random_seed: int,
    var_blob_shape: Sequence[int],
):
    rng = np.random.default_rng(random_seed)
    norm_conf = getattr(initializer_conf, "norm_conf")
    mean = getattr(norm_conf, "mean")
    std = getattr(norm_conf, "std")
    min = getattr(initializer_conf, "min")
    max = getattr(initializer_conf, "max")
    return lambda length: np.clip(
        rng.normal(loc=mean, scale=std, size=length), a_min=min, a_max=max
    )


@register_initializer("empty_conf")
def EmptyInitializerImpl(
    initializer_conf: initializer_conf_util.EmptyInitializerConf,
    random_seed: int,
    var_blob_shape: Sequence[int],
):
    return None


def get_initializer(initializer_conf, random_seed, var_blob_shape):
    f = None
    for m in _init_map:
        if initializer_conf.HasField(m):
            f = _init_map[m]
            break
    assert f is not None, initializer_conf
    return f(getattr(initializer_conf, m), random_seed, var_blob_shape)


def generate_values_by_initializer(initializer, shape, dtype):
    def elem_cnt(shape):
        return np.prod(shape).astype(int).item()

    np_dtype = np.dtype(dtype_util.convert_oneflow_dtype_to_numpy_dtype(dtype))
    length = elem_cnt(shape)
    return np.array(initializer(length)).astype(np_dtype).reshape(shape)

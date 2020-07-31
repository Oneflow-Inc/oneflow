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
from __future__ import absolute_import

import functools
import math

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.python.framework.dtype as dtype_util
from oneflow.python.oneflow_export import oneflow_export
from typing import Optional, Sequence


@oneflow_export("constant_initializer")
def constant_initializer(
    value: float = 0, dtype: dtype_util.dtype = dtype_util.float
) -> op_conf_util.InitializerConf:
    r"""Initializer that generates blob with constant values.
    
    Args:
        value: A Python scalar. All elements of the initialized variable 
        will be set to the corresponding value.
        dtype: Default data type.
    Returns:
        An InitializerConf object.
    """
    initializer = op_conf_util.InitializerConf()
    if dtype in [dtype_util.float, dtype_util.double]:
        setattr(initializer.constant_conf, "value", float(value))
    elif dtype in [
        dtype_util.int8,
        dtype_util.int32,
        dtype_util.int64,
    ]:
        setattr(initializer.constant_int_conf, "value", int(value))
    else:
        raise NotImplementedError("Do not support such data type")

    return initializer


@oneflow_export("zeros_initializer")
def zeros_initializer(
    dtype: dtype_util.dtype = dtype_util.float,
) -> op_conf_util.InitializerConf:
    return constant_initializer(0.0, dtype)


@oneflow_export("ones_initializer")
def ones_initializer(
    dtype: dtype_util.dtype = dtype_util.float,
) -> op_conf_util.InitializerConf:
    return constant_initializer(1.0, dtype)


@oneflow_export("random_uniform_initializer")
def random_uniform_initializer(
    minval: float = 0, maxval: float = 1, dtype: dtype_util.dtype = dtype_util.float
) -> op_conf_util.InitializerConf:
    r"""Initializer that generates blob with a uniform distribution.

    Args:
        minval: A python scalar. Lower bound of the range of random values to generate.
        maxval: A python scalar. Upper bound of the range of random values to generate. 
        Defaults to 1 for float types.
        seed: None. Not support yet.
        dtype: Default data type
    Returns:
        An InitializerConf object.
    """
    assert minval <= maxval
    initializer = op_conf_util.InitializerConf()
    if dtype in [dtype_util.float, dtype_util.double]:
        setattr(initializer.random_uniform_conf, "min", float(minval))
        setattr(initializer.random_uniform_conf, "max", float(maxval))
    elif dtype in [
        dtype_util.int8,
        dtype_util.int32,
        dtype_util.int64,
    ]:
        setattr(initializer.random_uniform_int_conf, "min", int(minval))
        setattr(initializer.random_uniform_int_conf, "max", int(maxval))
    else:
        raise NotImplementedError("Do not support such data type")

    return initializer


@oneflow_export("random_normal_initializer")
def random_normal_initializer(
    mean: float = 0.0,
    stddev: float = 1.0,
    seed: Optional[int] = None,
    dtype: Optional[dtype_util.dtype] = None,
) -> op_conf_util.InitializerConf:
    r"""Initializer that generates blob with a normal distribution.

    Args:
        mean: a python scalar. Mean of the random values to generate.
        stddev: a python scalar. Standard deviation of the random values to generate.
        seed: None. Not support yet.
        dtype: None. Not applicable in OneFlow
    Returns:
        An InitializerConf object.
    """
    assert seed is None
    assert dtype is None
    if seed is not None:
        assert name is not None
    initializer = op_conf_util.InitializerConf()
    setattr(initializer.random_normal_conf, "mean", float(mean))
    setattr(initializer.random_normal_conf, "std", float(stddev))

    return initializer


@oneflow_export("truncated_normal_initializer")
def truncated_normal_initializer(
    mean: float = 0.0, stddev: float = 1.0
) -> op_conf_util.InitializerConf:
    r"""Initializer that generates a truncated normal distribution.

    Args:
        mean: A scalar (float)
        stddev: A scalar (float)
    """
    initializer = op_conf_util.InitializerConf()
    setattr(initializer.truncated_normal_conf, "mean", float(mean))
    setattr(initializer.truncated_normal_conf, "std", float(stddev))
    return initializer


@oneflow_export("glorot_uniform_initializer", "xavier_uniform_initializer")
def glorot_uniform_initializer(data_format: str = "") -> op_conf_util.InitializerConf:
    return variance_scaling_initializer(1.0, "fan_avg", "random_uniform", data_format)


@oneflow_export("glorot_normal_initializer", "xavier_normal_initializer")
def glorot_normal_initializer(data_format: str = "") -> op_conf_util.InitializerConf:
    return variance_scaling_initializer(1.0, "fan_avg", "random_normal", data_format)


@oneflow_export("variance_scaling_initializer")
def variance_scaling_initializer(
    scale: float = 1.0,
    mode: str = "fan_in",
    distribution: str = "truncated_normal",
    data_format: str = "",
) -> op_conf_util.InitializerConf:
    r"""Initializer that generates a truncated normal distribution
    or a random normal distribution or a random uniform distribution
    with a scale adapting to it.

    Args:
        scale: Scaling factor (positive float).
        mode: One of "fan_in", "fan_out", "fan_avg".
        distribution: Random distribution to use. One of "truncated_normal",
            "random_normal", "random_uniform".
        data_format: A string be one of "N...C" or "NC..."
    """
    initializer = op_conf_util.InitializerConf()
    setattr(initializer.variance_scaling_conf, "scale", float(scale))
    setattr(
        initializer.variance_scaling_conf, "variance_norm", _get_variance_norm(mode),
    )
    setattr(
        initializer.variance_scaling_conf,
        "distribution",
        _get_random_distribution(distribution),
    )
    setattr(
        initializer.variance_scaling_conf, "data_format", _get_data_format(data_format),
    )
    return initializer


@oneflow_export("kaiming_initializer")
def kaiming_initializer(
    shape: Sequence[int],
    distribution: str = "random_normal",
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    negative_slope: float = 0.0,
    data_format: str = "NCHW",
) -> None:
    r"""Initialize weight according to the method described in `Delving deep into
    rectifiers: Surpassing human-level performance on ImageNet classification`
    - He, K. et al. (2015), using a normal or uniform distribution.

    Args:
        distribution: 'random_normal' or 'random_uniform'
        mode: 'fan_in', 'fan_out' or 'fan_avg'
        nonlinearity: None, 'tanh', 'sigmoid', 'relu' or 'leaky_relu'
        negative_slope: the negative slope of leaky_relu
        data_format: 'NCHW', 'NHWC'
    """
    assert isinstance(shape, tuple)
    # Kaiming Initialization only deals with FC, Conv and Deconv's weight
    assert len(shape) >= 2
    elem_cnt = functools.reduce(lambda a, b: a * b, shape, 1)
    assert elem_cnt > 0
    assert distribution in ["random_normal", "random_uniform"]
    assert mode in ["fan_in", "fan_out", "fan_avg"]
    assert nonlinearity in [None, "tanh", "sigmoid", "relu", "leaky_relu"]
    assert data_format in ["NCHW", "NHWC"]

    fan = _CalcFan(shape, mode, _get_data_format(data_format))
    gain = _CalcGain(nonlinearity, negative_slope)
    std = gain / math.sqrt(fan)
    if distribution == "random_normal":
        return flow.random_normal_initializer(0.0, std)
    elif distribution == "random_uniform":
        bound = math.sqrt(3.0) * std
        return flow.random_uniform_initializer(-bound, bound)
    else:
        raise NotImplementedError("Only support normal and uniform distribution")


def _get_variance_norm(mode):
    if mode.lower() == "fan_in":
        return op_conf_util.kFanIn
    elif mode.lower() == "fan_out":
        return op_conf_util.kFanOut
    elif mode.lower() == "fan_avg":
        return op_conf_util.kAverage
    else:
        raise ValueError("Invalid variance_norm")


def _get_random_distribution(distribution):
    if distribution.lower() == "truncated_normal":
        return op_conf_util.kTruncatedNormal
    elif distribution.lower() == "random_normal":
        return op_conf_util.kRandomNormal
    elif distribution.lower() == "random_uniform":
        return op_conf_util.kRandomUniform
    else:
        raise ValueError("Invalid random_distribution")


def _get_data_format(data_format):
    assert isinstance(data_format, str), "data_format must be a string"

    if data_format.startswith("NC"):
        return "channels_first"
    elif data_format.startswith("N") and data_format.endswith("C"):
        return "channels_last"
    else:
        assert data_format == "", ValueError(
            'data_format must be "N...C" or "NC..." or ""'
        )
        return ""


def _CalcFan(shape, mode, data_format):
    if len(shape) == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:  # Conv and Deconv
        fan_in = 1.0
        for dim in shape[1:]:
            fan_in *= dim
        fan_out = shape[0]
        if data_format == "channels_first":
            for dim in shape[2:]:
                fan_out *= dim
        elif data_format == "channels_last":
            for dim in shape[1:-1]:
                fan_out *= dim
        else:
            raise NotImplementedError(
                "Only support 'channels_first' and 'channels_last' data format"
            )

    if mode == "fan_avg":
        return (float(fan_in) + float(fan_out)) / 2
    elif mode == "fan_in":
        return float(fan_in)
    elif mode == "fan_out":
        return float(fan_out)
    else:
        raise NotImplementedError("Only support 'fan_in', 'fan_out' and 'fan_avg' mode")


def _CalcGain(nonlinearity, negative_slope):
    if nonlinearity is None or nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise NotImplementedError(
            "Only support None, 'tanh', 'sigmoid', 'relu' and 'leaky_relu' nonlinearity"
        )

from __future__ import absolute_import

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.common.data_type_pb2 as data_type_conf_util
from oneflow.python.oneflow_export import oneflow_export

import oneflow as flow
import math
import functools


@oneflow_export("constant_initializer")
def constant_initializer(value=0, dtype=data_type_conf_util.kFloat):
    initializer = op_conf_util.InitializerConf()
    if dtype in [data_type_conf_util.kFloat, data_type_conf_util.kDouble]:
        setattr(initializer.constant_conf, "value", float(value))
    elif dtype in [
        data_type_conf_util.kInt8,
        data_type_conf_util.kInt32,
        data_type_conf_util.kInt64,
    ]:
        setattr(initializer.constant_int_conf, "value", int(value))
    else:
        raise NotImplementedError("Do not support such data type")

    return initializer


@oneflow_export("zeros_initializer")
def zeros_initializer(dtype=data_type_conf_util.kFloat):
    return constant_initializer(0.0, dtype)


@oneflow_export("ones_initializer")
def ones_initializer(dtype=data_type_conf_util.kFloat):
    return constant_initializer(1.0, dtype)


@oneflow_export("random_uniform_initializer")
def random_uniform_initializer(
    minval=0, maxval=1, dtype=data_type_conf_util.kFloat
):
    initializer = op_conf_util.InitializerConf()
    if dtype in [data_type_conf_util.kFloat, data_type_conf_util.kDouble]:
        setattr(initializer.random_uniform_conf, "min", float(minval))
        setattr(initializer.random_uniform_conf, "max", float(maxval))
    elif dtype in [
        data_type_conf_util.kInt8,
        data_type_conf_util.kInt32,
        data_type_conf_util.kInt64,
    ]:
        setattr(initializer.random_uniform_int_conf, "min", int(minval))
        setattr(initializer.random_uniform_int_conf, "max", int(maxval))
    else:
        raise NotImplementedError("Do not support such data type")

    return initializer


@oneflow_export("random_normal_initializer")
def random_normal_initializer(mean=0.0, stddev=1.0):
    initializer = op_conf_util.InitializerConf()
    setattr(initializer.random_normal_conf, "mean", float(mean))
    setattr(initializer.random_normal_conf, "std", float(stddev))

    return initializer


@oneflow_export("truncated_normal_initializer")
def truncated_normal_initializer(mean=0.0, stddev=1.0):
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
def glorot_uniform_initializer(data_format=""):
    return variance_scaling_initializer(
        1.0, "fan_avg", "random_uniform", data_format
    )


@oneflow_export("variance_scaling_initializer")
def variance_scaling_initializer(
    scale=1.0, mode="fan_in", distribution="truncated_normal", data_format=""
):
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
        initializer.variance_scaling_conf,
        "variance_norm",
        _get_variance_norm(mode),
    )
    setattr(
        initializer.variance_scaling_conf,
        "distribution",
        _get_random_distribution(distribution),
    )
    setattr(
        initializer.variance_scaling_conf,
        "data_format",
        _get_data_format(data_format),
    )
    return initializer


@oneflow_export("kaiming_initializer")
def kaiming_initializer(
    shape,
    distribution="random_normal",
    mode="fan_in",
    nonlinearity="leaky_relu",
    negative_slope=0.0,
    data_format="channels_first",
):
    r"""Initialize weight according to the method described in `Delving deep into
    rectifiers: Surpassing human-level performance on ImageNet classification`
    - He, K. et al. (2015), using a normal or uniform distribution.

    Args:
        distribution: 'random_normal' or 'random_uniform'
        mode: 'fan_in', 'fan_out' or 'fan_avg'
        nonlinearity: None, 'tanh', 'sigmoid', 'relu' or 'leaky_relu'
        negative_slope: the negative slope of leaky_relu
        data_format: 'channels_first', 'channels_last'
    """
    assert isinstance(shape, tuple)
    # Kaiming Initialization only deals with FC, Conv and Deconv's weight
    assert len(shape) >= 2
    elem_cnt = functools.reduce(lambda a, b: a * b, shape, 1)
    assert elem_cnt > 0
    assert distribution in ["random_normal", "random_uniform"]
    assert mode in ["fan_in", "fan_out", "fan_avg"]
    assert nonlinearity in [None, "tanh", "sigmoid", "relu", "leaky_relu"]
    assert data_format in ["channels_first", "channels_last"]

    fan = _CalcFan(shape, mode, data_format)
    gain = _CalcGain(nonlinearity, negative_slope)
    std = gain / math.sqrt(fan)
    if distribution == "random_normal":
        return flow.random_normal_initializer(0.0, std)
    elif distribution == "random_uniform":
        bound = math.sqrt(3.0) * std
        return flow.random_uniform_initializer(-bound, bound)
    else:
        raise NotImplementedError(
            "Only support normal and uniform distribution"
        )


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
        raise NotImplementedError(
            "Only support 'fan_in', 'fan_out' and 'fan_avg' mode"
        )


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

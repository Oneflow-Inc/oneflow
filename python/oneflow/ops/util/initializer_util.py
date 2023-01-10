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
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.framework.dtype as dtype_util


def get_random_distribution(distribution):
    if distribution.lower() == "truncated_normal":
        return initializer_conf_util.kTruncatedNormal
    elif distribution.lower() == "random_normal":
        return initializer_conf_util.kRandomNormal
    elif distribution.lower() == "random_uniform":
        return initializer_conf_util.kRandomUniform
    else:
        raise ValueError("Invalid random_distribution")


def get_data_format(data_format):
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


def calc_fan(shape, mode, data_format):
    assert (
        len(shape) >= 2
    ), "Fan in and fan out can out be computed for tensor with fewer 2 dimensions"
    if len(shape) == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    else:
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
    if mode == "fan_sum":
        return float(fan_in) + float(fan_out)
    elif mode == "fan_in":
        return float(fan_in)
    elif mode == "fan_out":
        return float(fan_out)
    else:
        raise NotImplementedError("Only support 'fan_in', 'fan_out' and 'fan_sum' mode")


def calc_gain(nonlinearity, param=None):
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif (
            not isinstance(param, bool)
            and isinstance(param, int)
            or isinstance(param, float)
        ):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

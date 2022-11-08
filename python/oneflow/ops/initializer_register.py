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

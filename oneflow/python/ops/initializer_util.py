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

import numpy as np

import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
import oneflow.python.framework.dtype as dtype_util
from oneflow.python.oneflow_export import oneflow_export
from typing import Optional, Sequence, Union


@oneflow_export("constant_initializer")
def constant_initializer(
    value: float = 0, dtype: dtype_util.dtype = dtype_util.float
) -> initializer_conf_util.InitializerConf:
    r"""Initializer that generates blob with constant values.

    Args:
        value (float, optional): A Python scalar. All elements of the initialized variable . Defaults to 0.
        dtype (dtype_util.dtype, optional): Default data type. Defaults to dtype_util.float.

    Raises:
        NotImplementedError:  Do not support such data type.

    Returns:
        initializer_conf_util.InitializerConf:  An InitializerConf object.
    
    For example: 

    Example 1:

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def constant_Job() -> None:
            init = flow.constant_initializer(2.5)
            blob = flow.get_variable(
                "blob-weight",
                shape=(3, ),
                initializer=init,
                trainable=True
            )
            flow.watch(blob, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        constant_Job()

        # out [2.5 2.5 2.5]

    Example 2:

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_constant_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.constant_initializer(0.01)
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_constant_Job(x)

        # out.shape (1, 128, 32, 32)

    """
    initializer = initializer_conf_util.InitializerConf()
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
) -> initializer_conf_util.InitializerConf:
    r"""Initializer that generates blobs initialized to 0

    Args:
        dtype (dtype_util.dtype, optional): Default data type. Defaults to dtype_util.float.

    Returns:
        initializer_conf_util.InitializerConf: constant_initializer

    For example: 

    Example 1: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def zeros_Job() -> None:
            init = flow.zeros_initializer()
            blob = flow.get_variable(
                "blob-weight",
                shape=(3, ),
                initializer=init,
                trainable=True
            )
            flow.watch(blob, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        zeros_Job()

        # out [0. 0. 0.]

    Example 2: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_zero_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.zeros_initializer()
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_zero_Job(x)

        # out.shape (1, 128, 32, 32)

    """
    return constant_initializer(0.0, dtype)


@oneflow_export("ones_initializer")
def ones_initializer(
    dtype: dtype_util.dtype = dtype_util.float,
) -> initializer_conf_util.InitializerConf:
    r"""Initializer that generates blobs initialized to 1.

    Args:
        dtype (dtype_util.dtype, optional): Default data type. Defaults to dtype_util.float.

    Returns:
        initializer_conf_util.InitializerConf: constant_initializer

    For example: 

    Example 1: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def ones_Job() -> None:
            init = flow.ones_initializer()
            blob = flow.get_variable(
                "blob-weight",
                shape=(3, ),
                initializer=init,
                trainable=True
            )
            flow.watch(blob, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        ones_Job()

        # out [1. 1. 1.]

    Example 2: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_one_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.ones_initializer()
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_one_Job(x)
        
        # out.shape (1, 128, 32, 32)

    """
    return constant_initializer(1.0, dtype)


@oneflow_export("random_uniform_initializer")
def random_uniform_initializer(
    minval: float = 0, maxval: float = 1, dtype: dtype_util.dtype = dtype_util.float
) -> initializer_conf_util.InitializerConf:
    r"""Initializer that generates blobs with a uniform distribution. 

    Args:
        minval (float, optional): A python scalar. Lower bound of the range of random values to generate. Defaults to 0.
        maxval (float, optional): A python scalar. Upper bound of the range of random values to generate. Defaults to 1.
        dtype (dtype_util.dtype, optional): Default data type. Defaults to dtype_util.float.

    Raises:
        NotImplementedError: Do not support such data type.

    Returns:
        initializer_conf_util.InitializerConf:  Initial configuration

    For example: 

    Example 1: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def random_uniform_Job() -> None:
            init = flow.random_uniform_initializer(minval=0, maxval=0.5)
            blob = flow.get_variable(
                "blob-weight",
                shape=(3, ),
                initializer=init,
                trainable=True
            )
            flow.watch(blob, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        random_uniform_Job()

        # out [0.07557311 0.3943565  0.31875622]

    Example 2: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_random_uniform_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.random_uniform_initializer(minval=0, maxval=0.5)

            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_random_uniform_Job(x)
        
        # out.shape (1, 128, 32, 32)

    """
    assert minval <= maxval
    initializer = initializer_conf_util.InitializerConf()
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
) -> initializer_conf_util.InitializerConf:
    r"""Initializer that generates blob with a normal distribution.

    Args:
        mean (float, optional): A python scalar. Mean of the random values to generate.. Defaults to 0.0.
        stddev (float, optional): A python scalar. Standard deviation of the random values to generate. Defaults to 1.0.
        seed (Optional[int], optional): None. Not support yet. Defaults to None.
        dtype (Optional[dtype_util.dtype], optional): . Defaults to None.

    Returns:
        initializer_conf_util.InitializerConf: Initial configuration

    For example: 

    Example 1: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def random_normal_Job() -> None:
            init = flow.random_normal_initializer(mean=1, stddev=1)
            blob = flow.get_variable(
                "blob-weight",
                shape=(3, ),
                initializer=init,
                trainable=True
            )
            flow.watch(blob, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        random_normal_Job()

        # out [1.4190257 2.7663114 1.7114428]

    Example 2: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_random_normal_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.random_normal_initializer(mean=0, stddev=1)

            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_random_normal_Job(x)

        # out.shape (1, 128, 32, 32)

    """
    assert seed is None
    assert dtype is None
    if seed is not None:
        assert name is not None
    initializer = initializer_conf_util.InitializerConf()
    setattr(initializer.random_normal_conf, "mean", float(mean))
    setattr(initializer.random_normal_conf, "std", float(stddev))

    return initializer


@oneflow_export("truncated_normal_initializer")
def truncated_normal_initializer(
    mean: float = 0.0, stddev: float = 1.0
) -> initializer_conf_util.InitializerConf:
    r"""Initializer that generates a truncated normal distribution.

    Args:
        mean (float, optional): A scalar (float). Defaults to 0.0.
        stddev (float, optional): A scalar (float). Defaults to 1.0.

    Returns:
        initializer_conf_util.InitializerConf: Initial configuration

    For example: 

    Example 1: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def truncated_normal_Job() -> None:
            init = flow.truncated_normal_initializer(mean=1, stddev=1)
            blob = flow.get_variable(
                "blob-weight",
                shape=(3, ),
                initializer=init,
                trainable=True
            )
            flow.watch(blob, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        truncated_normal_Job()

        # out [1.8303236  0.09787154 0.83049864]

    Example 2: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_truncated_normal_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.truncated_normal_initializer(mean=0, stddev=1)

            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_truncated_normal_Job(x)

        # out.shape (1, 128, 32, 32)

    """
    initializer = initializer_conf_util.InitializerConf()
    setattr(initializer.truncated_normal_conf, "mean", float(mean))
    setattr(initializer.truncated_normal_conf, "std", float(stddev))
    return initializer


@oneflow_export("glorot_uniform_initializer", "xavier_uniform_initializer")
def glorot_uniform_initializer(
    data_format: str = "",
) -> initializer_conf_util.InitializerConf:
    r"""Initializer that generates a Xavier uniform distribution. 
    
    It also can be called as `oneflow.glorot_uniform_initializer`.  

    The equation is: 

    .. math:: 

        W\sim U(-\sqrt{\frac{{6}}{{n_j+n_{j+1}}}},\sqrt{\frac{{6}}{{n_j+n_{j+1}}}})

    :math:`U` means uniform distribution 

    :math:`n_j` means the amount of Nth layer parameters 

    Args:
        data_format (str, optional): The data format. Defaults to "".

    Returns:
        initializer_conf_util.InitializerConf: Initial configuration

    For example: 

    Example 1:

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def xavier_uniform_Job() -> None:
            init = flow.xavier_uniform_initializer()
            blob = flow.get_variable(
                "blob-weight",
                shape=(3, 3),
                initializer=init,
                trainable=True
            )
            flow.watch(blob, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        xavier_uniform_Job()

        # out [[-0.14424723 -0.9532095  -0.08723891]
        #      [-0.8011227  -0.29729813 -0.26769108]
        #      [ 0.9208976  -0.5971756  -0.15077025]]

    Example 2: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_xavier_uniform_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.xavier_uniform_initializer()
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_xavier_uniform_Job(x)

        # out.shape (1, 128, 32, 32)

    """
    return variance_scaling_initializer(1.0, "fan_avg", "random_uniform", data_format)


@oneflow_export("glorot_normal_initializer", "xavier_normal_initializer")
def glorot_normal_initializer(
    data_format: str = "",
) -> initializer_conf_util.InitializerConf:
    r"""Initializer that generates a Xavier normal distribution. 
    
    It also can be called as `oneflow.glorot_normal_initializer`.  

    The equation is: 

    .. math:: 

        W\sim N(0, \sqrt{\frac{{2}}{{n_j+n_{j+1}}}})

    :math:`N` means normal distribution 

    :math:`n_j` means the amount of Nth layer parameters 

    Args:
        data_format (str, optional): The data format. Defaults to "".

    Returns:
        initializer_conf_util.InitializerConf: Initial configuration

    For example: 

    Example 1: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def xavier_normal_Job() -> None:
            init = flow.xavier_normal_initializer()
            blob = flow.get_variable(
                "blob-weight",
                shape=(3, 3),
                initializer=init,
                trainable=True
            )
            flow.watch(blob, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        xavier_normal_Job()

        # out [[ 0.5908121  -0.10804518 -0.6148571 ]
        #      [ 1.4007381  -0.08172473  0.36579943]
        #      [-0.6461796  -0.15923311  0.33653972]]

    Example 2: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_xavier_normal_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.xavier_normal_initializer()
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_xavier_normal_Job(x)

        # out.shape (1, 128, 32, 32)

    """
    return variance_scaling_initializer(1.0, "fan_avg", "random_normal", data_format)


@oneflow_export("variance_scaling_initializer")
def variance_scaling_initializer(
    scale: float = 1.0,
    mode: str = "fan_in",
    distribution: str = "truncated_normal",
    data_format: str = "",
) -> initializer_conf_util.InitializerConf:
    r"""Initializer that generates a truncated normal distribution or a random normal distribution or a random uniform distribution with a scale adapting to it.

    When the distribution is "truncated_normal"

    The equation is: 

    .. math:: 

        W\sim N(0, \sqrt{\frac{{scale}}{{n}}})

    If mode is "fan_in", the "n" is the number of input units in the weight Blob. 

    If mode is "fan_out", the "n" is the number of output units in the weight Blob. 

    if mode is "fan_avg", the "n" is the average of the number of input and output units in the weight Blob

    Args:
        scale (float, optional): Scaling factor (positive float). Defaults to 1.0.
        mode (str, optional): One of "fan_in", "fan_out", "fan_avg". Defaults to "fan_in".
        distribution (str, optional): Random distribution to use. One of "truncated_normal",. Defaults to "truncated_normal".
        data_format (str, optional): A string be one of "N...C" or "NC...". Defaults to "".

    Returns:
        initializer_conf_util.InitializerConf: Initial configuration

    For example: 

    Example 1: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def variance_scale_Job() -> None:
            init = flow.variance_scaling_initializer(scale=2.0, mode="fan_avg")
            blob = flow.get_variable(
                "blob-weight",
                shape=(3, 3),
                initializer=init,
                trainable=True
            )
            flow.watch(blob, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        variance_scale_Job()

        # out [[-0.13931477  0.12266728 -0.9434968 ]
        #      [-0.49665168  0.10231158 -0.19194333]
        #      [-0.7902896  -1.7034698  -0.38695997]]

    Example 2: 

    .. code-block:: python 

        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_variance_scaling_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.variance_scaling_initializer(mode="fan_out")
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_variance_scaling_Job(x)

        # out.shape (1, 128, 32, 32)

    """
    initializer = initializer_conf_util.InitializerConf()
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

    When distribution is "random_normal"

    The equation is: 

    .. math:: 

        W \sim N(0, \sqrt{\frac{{2}}{{n}}})

    When distribution is "random_uniform"

    The equation is: 

    .. math:: 

        W \sim U(-\sqrt{\frac{{6}}{{n}}}, \sqrt{\frac{{6}}{{n}}})
    
    If mode is "fan_in", the "n" is the number of input units in the weight Blob. 

    If mode is "fan_out", the "n" is the number of output units in the weight Blob. 

    if mode is "fan_avg", the "n" is the average of the number of input and output units in the weight Blob

    Args:
        shape (Sequence[int]): Blob shape.
        distribution (str, optional): 'random_normal' or 'random_uniform'. Defaults to "random_normal".
        mode (str, optional): 'fan_in', 'fan_out' or 'fan_avg'. Defaults to "fan_in".
        nonlinearity (str, optional): None, 'tanh', 'sigmoid', 'relu' or 'leaky_relu'. Defaults to "leaky_relu".
        negative_slope (float, optional): The negative slope of leaky_relu. Defaults to 0.0.
        data_format (str, optional):  'NCHW', 'NHWC'. Defaults to "NCHW".

    Raises:
        NotImplementedError: Only support normal and uniform distribution

    Returns:
        [type]: flow.random_normal_initializer or flow.random_uniform_initializer

    For example: 

    Example 1: 

    .. code-block:: python 

        import oneflow as flow
        import oneflow.typing as tp


        def watch_handler(y: tp.Numpy):
            print("out", y)


        @flow.global_function()
        def kaiming_Job() -> None:
            init = flow.kaiming_initializer(shape=(3, 3), 
                                            mode="fan_avg", 
                                            nonlinearity="relu")
            blob = flow.get_variable(
                "blob-weight",
                shape=(3, 3),
                initializer=init,
                trainable=True
            )
            flow.watch(blob, watch_handler)


        checkpoint = flow.train.CheckPoint()
        checkpoint.init()
        kaiming_Job()

        # out [[ 0.54521346  0.32585594  1.3474437 ]
        #      [ 0.30729076 -0.19158769  0.2709008 ]
        #      [-0.95830524 -0.05093324  0.28178614]]

    Example 2: 

    .. code-block:: python 
    
        import oneflow as flow
        import numpy as np
        import oneflow.typing as tp


        @flow.global_function()
        def conv2d_kaiming_Job(x: tp.Numpy.Placeholder((1, 256, 32, 32))
        ) -> tp.Numpy:
            initializer = flow.kaiming_initializer(shape=(1, 256, 32, 32))
            conv2d = flow.layers.conv2d(
                x,
                filters=128,
                kernel_size=3,
                strides=1,
                padding='SAME',
                kernel_initializer=initializer, 
                name="Conv2d"
            )
            return conv2d


        x = np.random.randn(1, 256, 32, 32).astype(np.float32)
        out = conv2d_kaiming_Job(x)

        # out.shape (1, 128, 32, 32)

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
        return initializer_conf_util.kFanIn
    elif mode.lower() == "fan_out":
        return initializer_conf_util.kFanOut
    elif mode.lower() == "fan_avg":
        return initializer_conf_util.kAverage
    else:
        raise ValueError("Invalid variance_norm")


def _get_random_distribution(distribution):
    if distribution.lower() == "truncated_normal":
        return initializer_conf_util.kTruncatedNormal
    elif distribution.lower() == "random_normal":
        return initializer_conf_util.kRandomNormal
    elif distribution.lower() == "random_uniform":
        return initializer_conf_util.kRandomUniform
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


_init_map = {}


def register_initializer(flow_initializer):
    def deco(func):
        _init_map[flow_initializer] = func
        return func

    return deco


def GetInitializer(initializer_conf, random_seed, var_blob_shape):
    f = None
    for m in _init_map:
        if initializer_conf.HasField(m):
            f = _init_map[m]
            break
    assert f is not None, initializer_conf
    return f(getattr(initializer_conf, m), random_seed, var_blob_shape)


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


def RngTruncatedNormal(mean, std, length, rng):
    truncated_value = 2 * std
    data = np.empty(length)
    generated = 0
    ratio = 1.2
    while generated < length:
        remaining = length - generated
        norm = rng.normal(mean, std, size=int(remaining * ratio))
        truncated = norm[np.abs(norm - mean) < truncated_value][:remaining]
        data[generated : generated + len(truncated)] = truncated
        generated += len(truncated)
    return data


@register_initializer("truncated_normal_conf")
def TruncatedNormalInitializerImpl(
    initializer_conf: initializer_conf_util.TruncatedNormalInitializerConf,
    random_seed: int,
    var_blob_shape: Sequence[int],
):
    rng = np.random.default_rng(random_seed)
    return lambda length: RngTruncatedNormal(
        initializer_conf.mean, initializer_conf.std, length, rng,
    )


def GenInitialFan(initializer_conf, var_blob_shape: Sequence[int]):
    variance_norm = initializer_conf.variance_norm
    data_format = initializer_conf.data_format
    fan_in = np.prod(var_blob_shape[1:]).astype(np.int).item()
    fan_out = var_blob_shape[0]
    if data_format == "channel_first":
        fan_out *= np.prod(var_blob_shape[2:]).astype(np.int).item()
    else:
        fan_out *= np.prod(var_blob_shape[1:-1]).astype(np.int).item()

    if variance_norm == initializer_conf_util.kAverage:
        fan = (fan_in + fan_out) / 2
    elif variance_norm == initializer_conf_util.kFanIn:
        fan = fan_in
    elif variance_norm == initializer_conf_util.kFanOut:
        fan = fan_out
    else:
        raise NotImplemented()
    return fan


@register_initializer("variance_scaling_conf")
def VarianceScalingInitializerImpl(
    initializer_conf: initializer_conf_util.VarianceScalingInitializerConf,
    random_seed: int,
    var_blob_shape: Sequence[int],
):
    scale = initializer_conf.scale / GenInitialFan(initializer_conf, var_blob_shape)
    distribution = initializer_conf.distribution
    rng = np.random.default_rng(random_seed)
    if distribution == initializer_conf_util.kTruncatedNormal:
        stddev = math.sqrt(scale) / 0.87962566103423978
        return lambda length: RngTruncatedNormal(0, stddev, length, rng)
    elif distribution == initializer_conf_util.kRandomNormal:
        stddev = math.sqrt(scale)
        return lambda length: rng.normal(0, stddev, size=length,)
    elif distribution == initializer_conf_util.kRandomUniform:
        limit = math.sqrt(3.0 * scale)
        return lambda length: rng.uniform(low=-limit, high=limit, size=length)
    else:
        raise NotImplemented()

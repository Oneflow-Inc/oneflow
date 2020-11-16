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
import unittest
import collections
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

pool_confs = [
    {
        "x_shape": (1, 1, 10, 10),
        "ksize": 2,
        "strides": 1,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 3, 7, 7),
        "ksize": 3,
        "strides": 2,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 7, 7, 3),
        "ksize": 3,
        "strides": 2,
        "padding": "SAME",
        "data_format": "NHWC",
    },
    {
        "x_shape": (1, 5, 6, 6),
        "ksize": 3,
        "strides": 2,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 7, 5, 5),
        "ksize": 3,
        "strides": 2,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 3, 12, 12),
        "ksize": 2,
        "strides": 1,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 1, 11, 11),
        "ksize": 3,
        "strides": 2,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 10, 10, 1),
        "ksize": 3,
        "strides": 2,
        "padding": "SAME",
        "data_format": "NHWC",
    },
    {
        "x_shape": (1, 1, 10, 10, 10),
        "ksize": 2,
        "strides": 2,
        "padding": "VALID",
        "data_format": "NCDHW",
    },
    {
        "x_shape": (1, 7, 5, 5, 5),
        "ksize": 3,
        "strides": 1,
        "padding": "SAME",
        "data_format": "NCDHW",
    },
    {
        "x_shape": (1, 5, 5, 5, 7),
        "ksize": 3,
        "strides": 2,
        "padding": "VALID",
        "data_format": "NDHWC",
    },
    {
        "x_shape": (1, 3, 3, 3, 3),
        "ksize": 2,
        "strides": 1,
        "padding": "SAME",
        "data_format": "NCDHW",
    },
]


def _GetSequence(value, n, name):
    """Formats value from input"""
    if value is None:
        value = [1]
    elif not isinstance(value, collections.Sized):
        value = [value]

    current_n = len(value)
    if current_n == 1:
        return list(value * n)
    elif current_n == n:
        return list(value)
    else:
        raise ValueError(
            "{} should be of length 1 or {} but was {}".format(name, n, current_n)
        )


@flow.unittest.skip_unless_1n1d()
class TestPoolPadding(flow.unittest.TestCase):
    def test_pool(_):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["pool_conf"] = pool_confs
        arg_dict["data_type"] = ["float32"]
        arg_dict["pooling_type"] = ["AVG", "MAX"]
        arg_dict["is_dynamic"] = [True, False]

        for case in GenArgList(arg_dict):
            (device_type, pool_conf, data_type, pooling_type, is_dynamic) = case
            x_shape = pool_conf["x_shape"]
            ksize = pool_conf["ksize"]
            strides = pool_conf["strides"]
            padding = pool_conf["padding"]
            data_format = pool_conf["data_format"]
            flow.clear_default_session()

            # Random inputs
            x = np.random.randn(*x_shape).astype(type_name_to_np_type[data_type])
            dim = len(x.shape) - 2

            # TODO: these cases will fail in old implementation
            if dim == 3 and data_format == "NDHWC":
                continue
            # TF results
            with tf.GradientTape(persistent=True) as tape:
                x_tf = tf.Variable(x)
                strides = _GetSequence(strides, dim, "strides")
                pooling_f = None
                if pooling_type == "AVG":
                    pooling_f = getattr(tf.nn, "avg_pool{}d".format(dim))
                elif pooling_type == "MAX":
                    pooling_f = getattr(tf.nn, "max_pool{}d".format(dim))
                else:
                    raise ValueError("pooling_type must be AVG or MAX")
                y_tf = pooling_f(x_tf, ksize, strides, padding, data_format=data_format)

            dx_tf = tape.gradient(y_tf, x_tf, tf.constant(1.0, shape=y_tf.shape))

            def assert_grad(b):
                # TODO(hanbinbin): In eager mode, cannot derive b's is_dynamic correctly, therefore, using if .. else ...
                # Don't warry, is_dynamic will be removed in the next refactor and the problem will gone.
                if b.is_dynamic:
                    b_ndarray = b.numpy_list()[0]
                else:
                    b_ndarray = b.numpy()
                assert np.allclose(dx_tf.numpy(), b_ndarray), (
                    case,
                    dx_tf.numpy(),
                    b_ndarray,
                )

            # 1F results
            dtype = type_name_to_flow_type[data_type]

            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)

            tensor_def = None
            if is_dynamic:
                func_config.default_logical_view(flow.scope.mirrored_view())
                tensor_def = oft.ListNumpy.Placeholder
            else:
                tensor_def = oft.Numpy.Placeholder

            @flow.global_function(type="train", function_config=func_config)
            def pooling_job(x: tensor_def(x_shape, dtype=dtype)):
                v = flow.get_variable(
                    "x",
                    shape=x_shape,
                    dtype=dtype,
                    initializer=flow.constant_initializer(0),
                    trainable=True,
                )
                v = flow.cast_to_current_logical_view(v)
                flow.watch_diff(v, assert_grad)
                x += v
                with flow.scope.placement(device_type, "0:0"):
                    pooling_f = None
                    if pooling_type == "AVG":
                        pooling_f = getattr(flow.nn, "avg_pool{}d".format(dim))
                    elif pooling_type == "MAX":
                        pooling_f = getattr(flow.nn, "max_pool{}d".format(dim))
                    else:
                        raise ValueError("pooling_type must be AVG or MAX")

                    padding = pool_conf["padding"]
                    if padding == "SAME":
                        padding = "SAME_UPPER"
                    y = pooling_f(
                        x,
                        ksize=ksize,
                        strides=strides,
                        padding=padding,
                        data_format=data_format,
                    )
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
                ).minimize(y)
                return y

            if is_dynamic:
                x = [x]
            y = pooling_job(x).get()
            y_ndarray = None
            if is_dynamic:
                y_ndarray = y.numpy_list()[0]
            else:
                y_ndarray = y.numpy()
            assert y_ndarray.shape == y_tf.numpy().shape, (
                y_ndarray.shape,
                y_tf.numpy().shape,
            )
            assert np.allclose(y_ndarray, y_tf.numpy(), rtol=1e-5, atol=1e-5), (
                case,
                y_ndarray - y_tf.numpy(),
            )


if __name__ == "__main__":
    unittest.main()

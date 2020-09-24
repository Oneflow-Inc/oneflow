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


def test_layer_norm(_):
    confs = [
        {"x_shape": (4, 5, 2, 6), "begin_norm_axis": -1, "begin_params_axis": -1},
    ]
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["confs"] = confs
    arg_dict["data_type"] = ["float32"]
    arg_dict["trainable"] = [True, False]
    arg_dict["center"] = [True, False]
    arg_dict["scale"] = [True, False]
    arg_dict["epsilon"] = [1e-5, 1e-10]

    for case in GenArgList(arg_dict):
        (device_type, confs, data_type, trainable, center, scale, epsilon) = case
        x_shape = confs["x_shape"]
        begin_norm_axis = confs["begin_norm_axis"]
        begin_params_axis = confs["begin_params_axis"]
        flow.clear_default_session()
        assert (
            begin_norm_axis == begin_params_axis
        ), "tf doesn't support a dedicated begin_params_axis"
        # Random inputs
        x = np.random.randn(*x_shape).astype(type_name_to_np_type[data_type])
        dim = len(x.shape) - 2

        # TF results
        with tf.GradientTape(persistent=True) as tape:
            x_tf = tf.Variable(x)
            y_tf = tf.keras.layers.LayerNormalization(
                axis=begin_norm_axis,
                epsilon=epsilon,
                center=center,
                scale=scale,
                beta_initializer="zeros",
                gamma_initializer="ones",
                beta_regularizer=None,
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                trainable=trainable,
            )(x_tf)

        dx_tf = tape.gradient(y_tf, x_tf, tf.constant(1.0, shape=y_tf.shape))

        def assert_grad(b):
            diff = dx_tf.numpy() - b.numpy()
            max_diff = np.max(np.abs(diff))
            assert np.allclose(dx_tf.numpy(), b.numpy(), rtol=1e-5, atol=1e-5), (
                case,
                max_diff,
            )

        # 1F results
        dtype = type_name_to_flow_type[data_type]

        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)

        @flow.global_function(type="train", function_config=func_config)
        def test_job(x: oft.Numpy.Placeholder(x_shape, dtype=dtype)):
            v = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=dtype,
                initializer=flow.constant_initializer(0),
                trainable=True,
            )
            flow.watch_diff(v, assert_grad)
            x += v
            with flow.scope.placement(device_type, "0:0"):
                y = flow.layers.layer_norm(
                    x,
                    begin_norm_axis=begin_norm_axis,
                    begin_params_axis=begin_params_axis,
                    center=center,
                    scale=scale,
                )
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(y)
            return y

        check_point = flow.train.CheckPoint()
        check_point.init()
        y = test_job(x).get()
        assert y.numpy().shape == y_tf.numpy().shape, (
            y.numpy().shape,
            y_tf.numpy().shape,
        )
        diff = y.numpy() - y_tf.numpy()
        max_diff = np.max(np.abs(diff))
        assert np.allclose(y.numpy(), y_tf.numpy(), rtol=1e-5, atol=2e-3), (
            case,
            max_diff,
        )

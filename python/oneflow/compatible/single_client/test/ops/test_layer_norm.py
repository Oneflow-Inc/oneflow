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
import unittest
from collections import OrderedDict

import numpy as np
import tensorflow as tf
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


@flow.unittest.skip_unless_1n1d()
class TestLayerNorm(flow.unittest.TestCase):
    def test_layer_norm(_):
        confs = [
            {"x_shape": (40, 1023)},
            {"x_shape": (40, 1024)},
            {"x_shape": (40, 2047)},
            {"x_shape": (40, 2048)},
            {"x_shape": (40, 16384)},
        ]
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["confs"] = confs
        arg_dict["data_type"] = ["float32", "float16"]
        arg_dict["trainable"] = [True, False]
        arg_dict["center"] = [True, False]
        arg_dict["scale"] = [True, False]
        arg_dict["epsilon"] = [1e-05]
        arg_dict["fuse_add_to_output"] = [True, False]
        for case in GenArgList(arg_dict):
            (
                device_type,
                confs,
                data_type,
                trainable,
                center,
                scale,
                epsilon,
                fuse_add_to_output,
            ) = case
            if device_type == "cpu" and data_type == "float16":
                continue
            if device_type == "cpu" and fuse_add_to_output == True:
                continue
            x_shape = confs["x_shape"]
            if device_type == "cpu" and x_shape[1] != 1024:
                continue
            begin_norm_axis = 1
            begin_params_axis = 1
            flow.clear_default_session()
            assert (
                begin_norm_axis == begin_params_axis
            ), "tf doesn't support a dedicated begin_params_axis"
            if data_type == "float16":
                x = (
                    np.random.uniform(low=-1, high=1, size=x_shape)
                    .astype(np.float16)
                    .astype(np.float32)
                )
            else:
                x = np.random.uniform(low=-1, high=1, size=x_shape).astype(
                    type_name_to_np_type[data_type]
                )
            dim = len(x.shape) - 2
            with tf.GradientTape(persistent=True) as tape:
                x_tf = tf.Variable(x)
                if data_type == "float16":
                    x_tf = tf.cast(x_tf, dtype=tf.float16)
                    tf.keras.backend.set_floatx("float16")
                else:
                    tf.keras.backend.set_floatx("float32")
                layer = tf.keras.layers.LayerNormalization(
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
                )
                y_tf = layer(x_tf)
                y_tf = tf.math.sigmoid(y_tf)
                z_tf = y_tf + x_tf
            if data_type == "float16":
                dx_tf = tape.gradient(
                    z_tf, x_tf, tf.constant(1.0, shape=z_tf.shape, dtype=tf.float16)
                )
            else:
                dx_tf = tape.gradient(z_tf, x_tf, tf.constant(1.0, shape=z_tf.shape))
            grad = tape.gradient(z_tf, layer.trainable_variables)
            if trainable:
                if scale and center:
                    tf_gamma_diff = grad[0]
                    tf_beta_diff = grad[1]
                elif scale and (not center):
                    tf_gamma_diff = grad[0]
                elif not scale and center:
                    tf_beta_diff = grad[0]
                else:
                    pass
            else:
                pass

            def assert_grad(b):
                if data_type == "float16":
                    dx_of = b.numpy().astype(np.float16)
                    rtol = 0.001
                    atol = 0.05
                else:
                    dx_of = b.numpy()
                    rtol = 1e-5
                    atol = 1e-5
                diff = dx_tf.numpy() - dx_of
                max_diff = np.max(np.abs(diff))
                assert np.allclose(dx_tf.numpy(), dx_of, rtol=rtol, atol=atol), (
                    case,
                    max_diff,
                )

            def assert_grad_gamma(b):
                if data_type == "float16":
                    of_gamma_diff = b.numpy().astype(np.float16)
                    rtol = 0.001
                    atol = 0.05
                else:
                    of_gamma_diff = b.numpy()
                    rtol = 1e-5
                    atol = 1e-5
                diff = tf_gamma_diff.numpy() - of_gamma_diff
                max_diff = np.max(np.abs(diff))
                assert np.allclose(
                    tf_gamma_diff.numpy(), of_gamma_diff, rtol=rtol, atol=atol
                ), (case, max_diff)

            def assert_grad_beta(b):
                if data_type == "float16":
                    of_beta_diff = b.numpy().astype(np.float16)
                    rtol = 0.001
                    atol = 0.05
                else:
                    of_beta_diff = b.numpy()
                    rtol = 1e-5
                    atol = 1e-5
                diff = tf_beta_diff.numpy() - of_beta_diff
                max_diff = np.max(np.abs(diff))
                assert np.allclose(
                    tf_beta_diff.numpy(), of_beta_diff, rtol=rtol, atol=atol
                ), (case, max_diff)

            if data_type == "float16":
                dtype = flow.float
            else:
                dtype = type_name_to_flow_type[data_type]
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)
            func_config.enable_fuse_add_to_output(fuse_add_to_output)

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
                if data_type == "float16":
                    x = flow.cast(x, dtype=flow.float16)
                with flow.scope.placement(device_type, "0:0"):
                    param_shape = x.shape[begin_params_axis:]
                    gamma = None
                    beta = None
                    if center:
                        with flow.scope.namespace("LayerNorm"):
                            beta = flow.get_variable(
                                name="beta",
                                shape=param_shape,
                                dtype=flow.float,
                                initializer=flow.constant_initializer(0.0),
                                trainable=trainable,
                                model_name="beta",
                                reuse=False,
                            )
                            if trainable:
                                flow.watch_diff(beta, assert_grad_beta)
                            if data_type == "float16":
                                beta = flow.cast(beta, dtype=flow.float16)
                    if scale:
                        with flow.scope.namespace("LayerNorm"):
                            gamma = flow.get_variable(
                                name="gamma",
                                shape=param_shape,
                                dtype=flow.float,
                                initializer=flow.constant_initializer(1.0),
                                trainable=trainable,
                                model_name="gamma",
                                reuse=False,
                            )
                            if trainable:
                                if data_type == "float16":
                                    flow.watch_diff(gamma, assert_grad_gamma)
                                else:
                                    flow.watch_diff(gamma, assert_grad_gamma)
                            if data_type == "float16":
                                gamma = flow.cast(gamma, dtype=flow.float16)
                    x = flow.identity(x)
                    y = flow.nn.layer_norm(
                        x,
                        gamma=gamma,
                        beta=beta,
                        begin_norm_axis=begin_norm_axis,
                        begin_params_axis=begin_params_axis,
                        epsilon=epsilon,
                    )
                    y = flow.math.sigmoid(y)
                    z = y + x
                if data_type == "float16":
                    y = flow.cast(y, dtype=flow.float)
                    z = flow.cast(z, dtype=flow.float)
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [0.0001]), momentum=0
                ).minimize(z)
                return y

            y = test_job(x).get()
            if data_type == "float16":
                y_of = y.numpy().astype(np.float16)
            else:
                y_of = y.numpy()
            assert y_of.shape == y_tf.numpy().shape, (
                y_of.shape,
                y_tf.numpy().shape,
            )
            diff = y_of.astype(np.float16) - y_tf.numpy()
            max_diff = np.max(np.abs(diff))
            assert np.allclose(y_of, y_tf.numpy(), rtol=1e-05, atol=0.002), (
                case,
                max_diff,
            )


if __name__ == "__main__":
    unittest.main()

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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(
    device_type,
    a_shape,
    b_shape,
    transpose_a,
    transpose_b,
    data_type,
    fuse_add_to_output,
    enable_tf32,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.enable_fuse_add_to_output(fuse_add_to_output)
    flow.config.enable_tensor_float_32_compute(enable_tf32)
    if data_type == "float16":
        dtype = flow.float
    else:
        dtype = type_name_to_flow_type[data_type]

    @flow.global_function(type="train", function_config=func_config)
    def MatmulJob():
        with flow.scope.placement(device_type, "0:0"):
            a = flow.get_variable(
                "a",
                shape=a_shape,
                dtype=dtype,
                initializer=flow.random_uniform_initializer(minval=0, maxval=1),
                trainable=True,
            )
            b = flow.get_variable(
                "b",
                shape=b_shape,
                dtype=dtype,
                initializer=flow.random_uniform_initializer(minval=0, maxval=1),
                trainable=True,
            )
            if data_type == "float16":
                out = flow.matmul(
                    flow.cast(a, dtype=flow.float16),
                    flow.cast(b, dtype=flow.float16),
                    transpose_a,
                    transpose_b,
                )
                c = flow.get_variable(
                    "c",
                    shape=out.shape,
                    dtype=dtype,
                    initializer=flow.random_uniform_initializer(minval=-1, maxval=1),
                    trainable=True,
                )
                loss = flow.cast(
                    out + flow.cast(c, dtype=flow.float16), dtype=flow.float
                )
            else:
                out = flow.matmul(a, b, transpose_a, transpose_b)
                c = flow.get_variable(
                    "c",
                    shape=out.shape,
                    dtype=dtype,
                    initializer=flow.random_uniform_initializer(minval=-1, maxval=1),
                    trainable=True,
                )
                loss = out + c

            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(a, test_global_storage.Setter("a"))
            flow.watch_diff(a, test_global_storage.Setter("a_diff"))
            flow.watch(b, test_global_storage.Setter("b"))
            flow.watch_diff(b, test_global_storage.Setter("b_diff"))
            flow.watch(c, test_global_storage.Setter("c"))
            flow.watch_diff(c, test_global_storage.Setter("c_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    # OneFlow
    of_out = MatmulJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        a = tf.Variable(test_global_storage.Get("a"))
        b = tf.Variable(test_global_storage.Get("b"))
        c = tf.Variable(test_global_storage.Get("c"))
        if data_type == "float16":
            a = tf.cast(a, tf.float16)
            b = tf.cast(b, tf.float16)
            c = tf.cast(c, tf.float16)
        tf_out = tf.matmul(a, b, transpose_a, transpose_b)
        tf_out = tf_out + c
        if data_type == "float16":
            tf_out = tf.cast(tf_out, tf.float32)

    loss_diff = test_global_storage.Get("loss_diff")
    tf_a_diff = tape.gradient(tf_out, a, loss_diff)
    tf_b_diff = tape.gradient(tf_out, b, loss_diff)
    tf_c_diff = tape.gradient(tf_out, c, loss_diff)
    if data_type == "float16":
        tolerance = 2e-3
    else:
        tolerance = 1e-3
    assert np.allclose(
        of_out.numpy(), tf_out.numpy(), rtol=tolerance, atol=tolerance
    ), np.max(np.abs(of_out.numpy() - tf_out.numpy()))
    assert np.allclose(
        test_global_storage.Get("a_diff"),
        tf_a_diff.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )
    assert np.allclose(
        test_global_storage.Get("b_diff"),
        tf_b_diff.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )
    assert np.allclose(
        test_global_storage.Get("c_diff"),
        tf_c_diff.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )


def filter_args(arg_list):
    def trans_shape(shape):
        tmp_shape = shape[:-2]
        tmp_shape += (shape[-1], shape[-2])
        return tmp_shape

    ret = []
    for arg in arg_list:
        a_shape = arg[1]
        b_shape = arg[2]
        if arg[3]:  # transpose_a
            a_shape = trans_shape(a_shape)
        if arg[4]:  # transpose_b
            b_shape = trans_shape(b_shape)
        if a_shape[-1] == b_shape[-2]:
            ret.append(tuple(arg))
    return ret


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["a_shape"] = [(512, 256), (256, 512)]
    arg_dict["b_shape"] = [(256, 1024), (1024, 256)]
    arg_dict["transpose_a"] = [True, False]
    arg_dict["transpose_b"] = [True, False]
    arg_dict["data_type"] = ["float16", "float32", "double"]
    arg_dict["fuse_add_to_output"] = [True, False]
    arg_dict["enable_tf32"] = [True, False]
    matmul_args = filter_args(GenArgList(arg_dict))

    arg_dict.clear()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["a_shape"] = [(10, 10, 64, 32), (10, 10, 32, 64)]
    arg_dict["b_shape"] = [(10, 10, 32, 128), (10, 10, 128, 32)]
    arg_dict["transpose_a"] = [True, False]
    arg_dict["transpose_b"] = [True, False]
    arg_dict["data_type"] = ["float16", "float32", "double"]
    arg_dict["fuse_add_to_output"] = [True, False]
    arg_dict["enable_tf32"] = [True, False]
    batch_matmul_args = filter_args(GenArgList(arg_dict))
    return matmul_args + batch_matmul_args


@flow.unittest.skip_unless_1n1d()
class TestMatmul(flow.unittest.TestCase):
    def test_matmul(test_case):
        for arg in gen_arg_list():
            if arg[0] == "cpu" and (arg[5] == "float16" or arg[7] == True):
                continue
            if arg[5] != "float32" and arg[7] == True:
                continue
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()

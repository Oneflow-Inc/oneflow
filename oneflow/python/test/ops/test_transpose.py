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
import oneflow.typing as tp
import tensorflow as tf
import test_global_storage
from test_util import GenArgList


def compare_with_tensorflow(device_type, input_shape, perm):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def TransposeJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "input",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=2, maxval=5),
                trainable=True,
            )

            loss = flow.transpose(x, perm)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = TransposeJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        tf_out = tf.transpose(x, perm)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )


@flow.unittest.skip_unless_1n1d()
class TestTranspose(flow.unittest.TestCase):
    def test_transpose(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(10, 11, 12, 13)]
        arg_dict["perm"] = [(2, 0, 1, 3), (1, 0, 2, 3), (3, 2, 1, 0), (3, 1, 2, 0)]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_transpose2(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(10, 11, 12)]
        arg_dict["perm"] = [(2, 0, 1), (1, 0, 2), (2, 1, 0), (1, 2, 0)]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_transpose3(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(10, 11)]
        arg_dict["perm"] = [(1, 0), (0, 1)]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)

    def test_transpose4(test_case):
        # Test the param "batch_axis_non_change"

        @flow.global_function()
        def transpose_batchaxis_non_change_job(
            x: tp.Numpy.Placeholder((3, 3), dtype=flow.float, batch_axis=0),
        ) -> None:
            pre_batch_axis = x.batch_axis
            transpose_blob = flow.transpose(x, perm=[1, 0], batch_axis_non_change=True)
            tranposed_batch_axis = transpose_blob.batch_axis
            test_case.assertTrue(np.array_equal(pre_batch_axis, tranposed_batch_axis))

        x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).astype(np.float32)
        transpose_batchaxis_non_change_job(x)

    def test_transpose_dim6(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(2, 3, 4, 5, 6, 7)]
        arg_dict["perm"] = [(2, 0, 1, 3, 5, 4)]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()

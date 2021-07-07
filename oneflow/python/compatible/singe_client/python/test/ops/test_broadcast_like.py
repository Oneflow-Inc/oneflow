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
from test_util import GenArgList
import oneflow.typing as tp

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_broadcast_like_with_tf(
    device_type, input_shape, like_shape, broadcast_axes, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def broadcast_like_forward(
        x: tp.Numpy.Placeholder(shape=input_shape, dtype=flow.float),
        y: tp.Numpy.Placeholder(shape=like_shape, dtype=flow.float),
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.broadcast_like(x, y, broadcast_axes=broadcast_axes)

    x = np.random.rand(*input_shape).astype(np.float32)
    like = np.random.rand(*like_shape).astype(np.float32)
    of_out = broadcast_like_forward(x, like).get()
    np_out = np.broadcast_to(x, like_shape)
    assert np.allclose(of_out.numpy(), np_out, rtol=rtol, atol=atol)


@flow.unittest.skip_unless_1n1d()
class TestBroadcastLike(flow.unittest.TestCase):
    def test_broadcast_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["input_shape"] = [(5, 2)]
        arg_dict["like_shape"] = [(4, 5, 2)]
        arg_dict["broadcast_axes"] = [[0]]
        for arg in GenArgList(arg_dict):
            compare_broadcast_like_with_tf(*arg)

    def test_broadcast_like2(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["input_shape"] = [(5, 2)]
        arg_dict["like_shape"] = [(4, 6, 5, 2)]
        arg_dict["broadcast_axes"] = [[0, 1]]
        for arg in GenArgList(arg_dict):
            compare_broadcast_like_with_tf(*arg)

    def test_broadcast_like_grad(test_case):
        def watch_diff_handler(blob: tp.Numpy):
            test_case.assertTrue(np.array_equal(blob, [[3.0], [3.0], [3.0]]))

        @flow.global_function(type="train")
        def watch_matmul_diff_job(
            images: tp.Numpy.Placeholder((3, 3), dtype=flow.float),
        ) -> None:
            weight_initializer = flow.constant_initializer(2)
            weight_shape = (3, 1)
            weight = flow.get_variable(
                "three-weight", shape=weight_shape, initializer=weight_initializer
            )
            weight_broadcast = flow.broadcast_like(
                weight, like=images, broadcast_axes=(1,)
            )
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
            flow.optimizer.SGD(lr_scheduler, momentum=0.9).minimize(weight_broadcast)
            flow.watch_diff(weight, watch_diff_handler)

        x = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]).astype(np.float32)
        watch_matmul_diff_job(x)


if __name__ == "__main__":
    unittest.main()

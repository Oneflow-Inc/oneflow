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
import numpy as np
import tensorflow as tf
import oneflow as flow
import oneflow.typing as oft
from collections import OrderedDict

from test_util import GenArgList
import test_global_storage
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, data_type, shape):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    dtype = type_name_to_flow_type[data_type]

    def np_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @flow.global_function(type="train", function_config=func_config)
    def SigmoidCrossEntropyWithLogitsJob(labels: oft.Numpy.Placeholder(shape, dtype)):
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=shape,
                dtype=type_name_to_flow_type[data_type],
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = flow.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=x)

            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))
            return loss

    # fake labels
    labels = np_sigmoid(np.random.randint(0, 10, size=shape)).astype(
        type_name_to_np_type[data_type]
    )

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = SigmoidCrossEntropyWithLogitsJob(labels).get()

    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        tf_out = tf.nn.sigmoid_cross_entropy_with_logits(labels, x)

        loss_diff = test_global_storage.Get("loss_diff")
        tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    tolerance = 1e-5
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=tolerance, atol=tolerance)
    assert np.allclose(
        test_global_storage.Get("x_diff"),
        tf_x_diff.numpy(),
        rtol=tolerance,
        atol=tolerance,
    )
    flow.clear_default_session()


@flow.unittest.skip_unless_1n1d()
class TestSigmoidCrossEntropy(flow.unittest.TestCase):
    def test_sigmoid_cross_entropy_with_logits(test_case):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["data_type"] = ["double", "float32"]
        arg_dict["shape"] = [(64, 1000), (5, 5, 1000)]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()

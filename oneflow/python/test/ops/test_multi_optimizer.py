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
import numpy as np
import oneflow as flow
import tensorflow as tf
from collections import OrderedDict

from test_util import GenArgList

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_multi_optimizer_with_tensorflow(
    device_type, x_shape, momentum, learning_rate, train_iters
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def testMultiSGD():
        with flow.scope.placement(device_type, "0:0-0"):
            x = flow.get_variable(
                name="x",
                shape=x_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            y = flow.get_variable(
                name="y",
                shape=x_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            loss = flow.math.reduce_sum(x + y)
            opt1 = flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                momentum=momentum,
                variables=["x"],
            )
            opt2 = flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                momentum=momentum,
                variables=["y"],
            )
            flow.optimizer.CombinedOptimizer([opt1, opt2]).minimize(loss)
            return (x, y)

    init_x = None
    init_y = None
    x = None
    y = None
    for i in range(train_iters + 1):
        x, y = testMultiSGD().get()
        if i == 0:
            init_x = np.copy(x.numpy())
            init_y = np.copy(y.numpy())

    var_x = tf.Variable(init_x)
    var_y = tf.Variable(init_y)
    opt1 = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=momentum, nesterov=False
    )
    opt2 = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=momentum, nesterov=False
    )

    for i in range(train_iters):
        with tf.GradientTape(persistent=True) as tape:
            loss = tf.math.reduce_sum(var_x + var_y)

        grad_x = tape.gradient([loss], var_x)
        grad_y = tape.gradient([loss], var_y)

        opt1.apply_gradients([(grad_x, var_x)])
        opt2.apply_gradients([(grad_y, var_y)])

    print(x.numpy())
    print(x.numpy())
    print(var_x.numpy())
    print(var_y.numpy())

    print(x.flatten() - var_x.numpy().flatten())
    print(y.flatten() - var_y.numpy().flatten())

    assert np.allclose(x.flatten(), var_x.numpy().flatten(), rtol=1e-4, atol=1e-4,)
    assert np.allclose(y.flatten(), var_y.numpy().flatten(), rtol=1e-4, atol=1e-4,)


@flow.unittest.skip_unless_1n1d()
class TestOptimizers(flow.unittest.TestCase):
    def test_multi_sgd(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["momentum"] = [0.0, 0.9]
        arg_dict["learning_rate"] = [1]
        arg_dict["train_iters"] = [10]
        for arg in GenArgList(arg_dict):
            compare_multi_optimizer_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()

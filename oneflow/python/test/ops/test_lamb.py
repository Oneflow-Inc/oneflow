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
import tensorflow_addons as tfa
import test_global_storage
from test_util import GenArgList


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow_addons_lamb(
    test_case,
    device_type,
    x_shape,
    beta1,
    beta2,
    epsilon,
    weight_decay,
    learning_rate,
    train_iters,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=flow.FunctionConfig())
    def testLAMB(
        random_mask: flow.typing.Numpy.Placeholder(x_shape, dtype=flow.float32)
    ) -> flow.typing.Numpy:
        with flow.scope.placement(device_type, "0:0-0"):
            x = flow.get_variable(
                name="x",
                shape=x_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = flow.math.reduce_mean(x + random_mask)
            flow.optimizer.LAMB(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                weight_decay=weight_decay,
            ).minimize(loss)
            return x

    # generate random number sequences
    random_masks_seq = []
    for i in range(train_iters + 1):
        random_masks_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    x_list = []
    init_value = None
    for i in range(train_iters + 1):
        x = testLAMB(random_masks_seq[i])
        x_list.append(x)
        if i == 0:
            init_value = np.copy(x)

    var = tf.Variable(init_value)
    opt = tfa.optimizers.LAMB(
        learning_rate=learning_rate,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon,
        weight_decay_rate=weight_decay,
    )

    var_list = []
    for i in range(train_iters):
        with tf.GradientTape() as tape:
            if i == 0:
                var0 = tf.identity(var)
                var_list.append(var0)
            random_mask = tf.Variable(random_masks_seq[i])
            loss = tf.reduce_mean(var + random_mask)
        gradients = tape.gradient(loss, var)
        opt.apply_gradients(zip([gradients], [var]))
        var_list.append(var.numpy())
    case = (
        device_type,
        x_shape,
        beta1,
        beta2,
        epsilon,
        weight_decay,
        learning_rate,
        train_iters,
    )
    test_case.assertTrue(len(x_list) == len(var_list))
    for (i, o, t) in zip(range(len(var_list)), x_list, var_list):
        diff = o - t
        test_case.assertTrue(
            np.allclose(x_list[i], var_list[i], rtol=1e-3, atol=1e-3), (i, case, diff),
        )
    diff = x.flatten() - var.numpy().flatten()
    test_case.assertTrue(
        np.allclose(x.flatten(), var.numpy().flatten(), rtol=1e-3, atol=1e-3),
        (case, diff),
    )


@flow.unittest.skip_unless_1n1d()
class TestLamb(flow.unittest.TestCase):
    def test_lamb(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["x_shape"] = [(10,)]
        arg_dict["beta1"] = [0.9]
        arg_dict["beta2"] = [0.999]
        arg_dict["epsilon"] = [1e-6]
        arg_dict["weight_decay"] = [0.01]
        arg_dict["learning_rate"] = [1e-4]
        arg_dict["train_iters"] = [10]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow_addons_lamb(*arg)


if __name__ == "__main__":
    unittest.main()

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
    device_type,
    var1_shape,
    var2_shape,
    var3_shape,
    sgd_opt_args,
    rmsprop_opt_args,
    adam_opt_args,
    train_iters,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def TestMultiOptimizerJob():
        with flow.scope.placement(device_type, "0:0-0"):
            var1 = flow.get_variable(
                name="var1",
                shape=var1_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            var2 = flow.get_variable(
                name="var2",
                shape=var2_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            var3 = flow.get_variable(
                name="var3",
                shape=var3_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            loss = flow.math.reduce_sum(var1 + var2 + var3)
            sgd_opt = flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [sgd_opt_args["lr"]]),
                momentum=sgd_opt_args["momentum"],
                variables=["var1"],
            )
            rmsprop_opt = flow.optimizer.RMSProp(
                flow.optimizer.PiecewiseConstantScheduler([], [rmsprop_opt_args["lr"]]),
                decay_rate=rmsprop_opt_args["decay_rate"],
                epsilon=0,
                centered=rmsprop_opt_args["centered"],
                variables=["var2"],
            )
            adam_opt = flow.optimizer.Adam(
                flow.optimizer.PiecewiseConstantScheduler([], [adam_opt_args["lr"]]),
                beta1=adam_opt_args["beta1"],
                beta2=adam_opt_args["beta2"],
                epsilon=adam_opt_args["epsilon"],
                do_bias_correction=True,
                variables=["var3"],
            )
            flow.optimizer.CombinedOptimizer([sgd_opt, rmsprop_opt, adam_opt]).minimize(
                loss
            )
            return (var1, var2, var3)

    init_var1 = None
    init_var2 = None
    init_var3 = None
    for i in range(train_iters + 1):
        var1, var2, var3 = TestMultiOptimizerJob().get()
        if i == 0:
            init_var1 = np.copy(var1.numpy())
            init_var2 = np.copy(var2.numpy())
            init_var3 = np.copy(var3.numpy())

    tf_var1 = tf.Variable(init_var1)
    tf_var2 = tf.Variable(init_var2)
    tf_var3 = tf.Variable(init_var3)
    tf_sgd_opt = tf.keras.optimizers.SGD(
        learning_rate=sgd_opt_args["lr"],
        momentum=sgd_opt_args["momentum"],
        nesterov=False,
    )
    tf_rmsprop_opt = tf.keras.optimizers.RMSprop(
        learning_rate=rmsprop_opt_args["lr"],
        rho=rmsprop_opt_args["decay_rate"],
        momentum=0.0,
        epsilon=0,
        centered=rmsprop_opt_args["centered"],
    )
    tf_adam_opt = tf.keras.optimizers.Adam(
        learning_rate=adam_opt_args["lr"],
        beta_1=adam_opt_args["beta1"],
        beta_2=adam_opt_args["beta2"],
        epsilon=adam_opt_args["epsilon"],
        amsgrad=False,
    )

    for i in range(train_iters):
        with tf.GradientTape(persistent=True) as tape:
            loss = tf.math.reduce_sum(tf_var1 + tf_var2 + tf_var3)

        tf_var1_grad = tape.gradient([loss], tf_var1)
        tf_var2_grad = tape.gradient([loss], tf_var2)
        tf_var3_grad = tape.gradient([loss], tf_var3)

        tf_sgd_opt.apply_gradients([(tf_var1_grad, tf_var1)])
        tf_rmsprop_opt.apply_gradients([(tf_var2_grad, tf_var2)])
        tf_adam_opt.apply_gradients([(tf_var3_grad, tf_var3)])

    assert np.allclose(var1.flatten(), tf_var1.numpy().flatten(), rtol=1e-4, atol=1e-4,)
    assert np.allclose(var2.flatten(), tf_var2.numpy().flatten(), rtol=5e-3, atol=5e-3,)
    assert np.allclose(var3.flatten(), tf_var3.numpy().flatten(), rtol=1e-4, atol=1e-4,)


@flow.unittest.skip_unless_1n1d()
class TestMultiOptimizer(flow.unittest.TestCase):
    def test_multi_optimizer(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["var1_shape"] = [(10,)]
        arg_dict["var2_shape"] = [(10,)]
        arg_dict["var3_shape"] = [(10,)]
        arg_dict["sgd_opt_args"] = [{"lr": 1, "momentum": 0.9}]
        arg_dict["rmsprop_opt_args"] = [
            {"lr": 0.5, "decay_rate": 0.9, "centered": False}
        ]
        arg_dict["adam_opt_args"] = [
            {"lr": 2, "beta1": 0.9, "beta2": 0.99, "epsilon": 1e-9}
        ]
        arg_dict["train_iters"] = [10]
        for arg in GenArgList(arg_dict):
            compare_multi_optimizer_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()

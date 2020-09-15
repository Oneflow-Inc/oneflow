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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
import test_global_storage
from test_util import GenArgList


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow_rmsprop(
    device_type, x_shape, centered, decay_rate, learning_rate, train_iters
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def testRmsprop(
        random_mask: flow.typing.Numpy.Placeholder(x_shape, dtype=flow.float32)
    ) -> flow.typing.Numpy:
        with flow.scope.placement(device_type, "0:0-0"):
            x = flow.get_variable(
                name="x",
                shape=x_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            loss = flow.math.reduce_mean(x * random_mask)
            flow.optimizer.RMSProp(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                decay_rate=decay_rate,
                epsilon=0,
                centered=centered,
            ).minimize(loss)
            return x

    checkpoint = flow.train.CheckPoint()
    checkpoint.init()

    # generate random number sequences
    random_masks_seq = []
    for i in range(train_iters + 1):
        random_masks_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    init_value = None
    for i in range(train_iters + 1):
        x = testRmsprop(random_masks_seq[i])
        if i == 0:
            init_value = np.copy(x)

    var = tf.Variable(init_value)
    opt = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate,
        rho=decay_rate,
        momentum=0.0,
        epsilon=0,
        centered=centered,
    )

    for i in range(train_iters):
        with tf.GradientTape() as tape:
            random_mask = tf.Variable(random_masks_seq[i])
            loss = tf.reduce_mean(var * random_mask)
        gradients = tape.gradient(loss, var)
        opt.apply_gradients(zip([gradients], [var]))

    assert np.allclose(x.flatten(), var.numpy().flatten(), rtol=1e-3, atol=1e-3,), (
        x.flatten() - var.numpy().flatten()
    )


def compare_with_tensorflow_adam(
    device_type, x_shape, beta1, beta2, epsilon, learning_rate, train_iters
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=flow.FunctionConfig())
    def testAdam(
        random_mask: flow.typing.Numpy.Placeholder(x_shape, dtype=flow.float32)
    ) -> flow.typing.Numpy:
        with flow.scope.placement(device_type, "0:0-0"):
            x = flow.get_variable(
                name="x",
                shape=x_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            loss = flow.math.reduce_mean(x * random_mask)
            flow.optimizer.Adam(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                do_bias_correction=True,
            ).minimize(loss)
            return x

    checkpoint = flow.train.CheckPoint()
    checkpoint.init()

    # generate random number sequences
    random_masks_seq = []
    for i in range(train_iters + 1):
        random_masks_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    init_value = None
    for i in range(train_iters + 1):
        x = testAdam(random_masks_seq[i])
        if i == 0:
            init_value = np.copy(x)

    var = tf.Variable(init_value)
    opt = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon,
        amsgrad=False,
    )

    for i in range(train_iters):
        with tf.GradientTape() as tape:
            random_mask = tf.Variable(random_masks_seq[i])
            loss = tf.reduce_mean(var * random_mask)
        gradients = tape.gradient(loss, var)
        opt.apply_gradients(zip([gradients], [var]))

    assert np.allclose(x.flatten(), var.numpy().flatten(), rtol=1e-4, atol=1e-4,)


def compare_with_numpy_adamw(
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
    def testAdamW(
        random_mask: flow.typing.Numpy.Placeholder(x_shape, dtype=flow.float32)
    ) -> flow.typing.Numpy:
        with flow.scope.placement(device_type, "0:0-0"):
            x = flow.get_variable(
                name="x",
                shape=x_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            loss = flow.math.reduce_mean(x * random_mask)
            flow.optimizer.AdamW(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                weight_decay=weight_decay,
                do_bias_correction=True,
            ).minimize(loss)
            return x

    checkpoint = flow.train.CheckPoint()
    checkpoint.init()

    # generate random number sequences
    random_masks_seq = []
    for i in range(train_iters + 1):
        random_masks_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    init_value = None
    for i in range(train_iters + 1):
        x = testAdamW(random_masks_seq[i])
        if i == 0:
            init_value = np.copy(x)

    def adamw_update_numpy(
        param,
        gradient,
        iter,
        m,
        v,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        weight_decay=0.9,
    ):
        lr_t = lr * np.sqrt(1 - beta2 ** (iter + 1)) / (1 - beta1 ** (iter + 1))

        m_t = beta1 * m + (1 - beta1) * gradient
        v_t = beta2 * v + (1 - beta2) * gradient * gradient

        param_t = param - lr_t * (m_t / (np.sqrt(v_t) + epsilon) + weight_decay * param)
        return param_t, m_t, v_t

    param = init_value
    gradient = np.full(param.shape, 1.0 / np.prod(param.shape))
    m = np.zeros(param.shape)
    v = np.zeros(param.shape)
    for i in range(train_iters):
        param, m, v = adamw_update_numpy(
            param,
            gradient * random_masks_seq[i],
            i,
            m,
            v,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
        )

    assert np.allclose(x.flatten(), param.flatten(), rtol=1e-4, atol=1e-4,)


def compare_with_numpy_lazy_adam(
    device_type, x_shape, beta1, beta2, epsilon, learning_rate, train_iters,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def testLazyAdam(
        mask: flow.typing.Numpy.Placeholder(x_shape, dtype=flow.float32)
    ) -> flow.typing.Numpy:
        with flow.scope.placement(device_type, "0:0-0"):
            x = flow.get_variable(
                name="x",
                shape=x_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            loss = flow.math.reduce_mean(x * mask)

            flow.optimizer.LazyAdam(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
            ).minimize(loss)

            return x

    checkpoint = flow.train.CheckPoint()
    checkpoint.init()

    mask = np.random.randint(2, size=x_shape)
    mask_float = mask.astype(np.float32)

    init_value = None
    for i in range(train_iters + 1):
        x = testLazyAdam(mask_float)
        if i == 0:
            init_value = np.copy(x)

    def lazy_adam_update_numpy(
        param,
        mask,
        gradient,
        iter,
        m,
        v,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
    ):

        lr_t = lr * np.sqrt(1 - beta2 ** (iter + 1)) / (1 - beta1 ** (iter + 1))

        m_t = np.copy(m)
        v_t = np.copy(v)

        m_t_o = beta1 * m + (1 - beta1) * gradient
        v_t_o = beta2 * v + (1 - beta2) * gradient * gradient

        m_t[mask == 1] = m_t_o[mask == 1]
        v_t[mask == 1] = v_t_o[mask == 1]

        param_t = np.copy(param)

        param_t_o = param - lr_t * m_t / (np.sqrt(v_t) + epsilon)

        param_t[mask == 1] = param_t_o[mask == 1]

        return param_t, m_t, v_t

    param = init_value
    gradient = np.full(param.shape, 1.0 / np.prod(param.shape))
    m = np.zeros(param.shape)
    v = np.zeros(param.shape)

    for i in range(train_iters):
        param, m, v = lazy_adam_update_numpy(
            param, mask, gradient, i, m, v, learning_rate, beta1, beta2, epsilon
        )

    assert np.allclose(x.flatten(), param.flatten(), rtol=1e-4, atol=1e-4,)


def compare_with_numpy_lars(
    device_type,
    x_shape,
    momentum_beta,
    epsilon,
    lars_coefficient,
    learning_rate,
    train_iters,
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def testLars(
        random_mask: flow.typing.Numpy.Placeholder(x_shape, dtype=flow.float32)
    ) -> flow.typing.Numpy:
        with flow.scope.placement(device_type, "0:0-0"):
            x = flow.get_variable(
                name="x",
                shape=x_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            loss = flow.math.reduce_mean(x * random_mask)

            flow.optimizer.LARS(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                momentum_beta=momentum_beta,
                epsilon=epsilon,
                lars_coefficient=lars_coefficient,
            ).minimize(loss)
            return x

    checkpoint = flow.train.CheckPoint()
    checkpoint.init()

    # generate random number sequences
    random_masks_seq = []
    for i in range(train_iters + 1):
        random_masks_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    init_value = None
    for i in range(train_iters + 1):
        x = testLars(random_masks_seq[i])
        if i == 0:
            init_value = np.copy(x)

    def lars_update_numpy(
        param,
        gradient,
        iter,
        momentum,
        learning_rate=0.001,
        momentum_beta=0.9,
        epsilon=1e-9,
        lars_coefficient=0.0001,
    ):
        import math

        model_norm = math.sqrt(np.mean(param * param))
        model_diff_norm = math.sqrt(np.mean(gradient * gradient))

        local_learning_rate = (
            learning_rate * lars_coefficient * model_norm / (epsilon + model_diff_norm)
        )

        momentum_t = momentum_beta * momentum - local_learning_rate * gradient

        param_t = param + momentum_t

        return param_t, momentum_t

    param = init_value
    gradient = np.full(param.shape, 1.0 / np.prod(param.shape))
    momentum = np.zeros(param.shape)

    for i in range(train_iters):
        param, momentum = lars_update_numpy(
            param,
            gradient * random_masks_seq[i],
            i,
            momentum,
            learning_rate,
            momentum_beta,
            epsilon,
            lars_coefficient,
        )

    assert np.allclose(x.flatten(), param.flatten(), rtol=1e-4, atol=1e-4,)


def compare_with_tensorflow_sgd(
    device_type, x_shape, momentum, learning_rate, train_iters
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=flow.FunctionConfig())
    def testSGD(
        random_mask: flow.typing.Numpy.Placeholder(x_shape, dtype=flow.float32)
    ) -> flow.typing.Numpy:
        with flow.scope.placement(device_type, "0:0-0"):
            x = flow.get_variable(
                name="x",
                shape=x_shape,
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            loss = flow.math.reduce_mean(x * random_mask)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                momentum=momentum,
            ).minimize(loss)
            return x

    checkpoint = flow.train.CheckPoint()
    checkpoint.init()

    # generate random number sequences
    random_masks_seq = []
    for i in range(train_iters + 1):
        random_masks_seq.append(np.random.uniform(size=x_shape).astype(np.float32))

    init_value = None
    for i in range(train_iters + 1):
        x = testSGD(random_masks_seq[i])
        if i == 0:
            init_value = np.copy(x)

    var = tf.Variable(init_value)
    opt = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=momentum, nesterov=False
    )

    for i in range(train_iters):
        with tf.GradientTape() as tape:
            random_mask = tf.Variable(random_masks_seq[i])
            loss = tf.reduce_mean(var * random_mask)
        gradients = tape.gradient(loss, var)
        opt.apply_gradients(zip([gradients], [var]))

    assert np.allclose(x.flatten(), var.numpy().flatten(), rtol=1e-4, atol=1e-4,)


def test_rmsprop(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(10,)]
    arg_dict["centered"] = [True, False]
    arg_dict["decay_rate"] = [0.9]
    arg_dict["learning_rate"] = [1]
    arg_dict["train_iters"] = [10]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow_rmsprop(*arg)


def test_adam(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(10,)]
    arg_dict["beta1"] = [0.9]
    arg_dict["beta2"] = [0.99]
    arg_dict["epsilon"] = [1e-9]
    arg_dict["learning_rate"] = [1]
    arg_dict["train_iters"] = [10]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow_adam(*arg)


def test_lazy_adam(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(10,)]
    arg_dict["beta1"] = [0.9]
    arg_dict["beta2"] = [0.99]
    arg_dict["epsilon"] = [1e-9]
    arg_dict["learning_rate"] = [1]
    arg_dict["train_iters"] = [10]
    for arg in GenArgList(arg_dict):
        compare_with_numpy_lazy_adam(*arg)


def test_adamw(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(10,)]
    arg_dict["beta1"] = [0.9]
    arg_dict["beta2"] = [0.99]
    arg_dict["epsilon"] = [1e-9]
    arg_dict["weight_decay"] = [0.9]
    arg_dict["learning_rate"] = [1]
    arg_dict["train_iters"] = [10]
    for arg in GenArgList(arg_dict):
        compare_with_numpy_adamw(*arg)


def test_lars(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(10,)]
    arg_dict["momentum_beta"] = [0.9]
    arg_dict["epsilon"] = [1e-9]
    arg_dict["lars_coefficient"] = [0.0001]
    arg_dict["learning_rate"] = [1]
    arg_dict["train_iters"] = [10]
    for arg in GenArgList(arg_dict):
        compare_with_numpy_lars(*arg)


def test_sgd(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["x_shape"] = [(10,)]
    arg_dict["momentum"] = [0.9, 0.0]
    arg_dict["learning_rate"] = [1]
    arg_dict["train_iters"] = [10]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow_sgd(*arg)

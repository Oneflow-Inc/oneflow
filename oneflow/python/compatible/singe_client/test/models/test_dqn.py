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
import oneflow as flow
import oneflow.typing as oft
import numpy as np
import os
import unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
def test_1n1c(test_case):
    dqn = DQN("gpu")
    dqn.test_parameters_copy()


# get QNet parameters
def getQNetParams(var_name_prefix: str = "QNet", is_train: bool = True):
    weight_init = flow.variance_scaling_initializer(
        scale=1.0, mode="fan_in", distribution="truncated_normal", data_format="NCHW"
    )
    bias_init = flow.constant_initializer(value=0.0)

    conv_prefix = "_conv1"
    conv1_weight = flow.get_variable(
        var_name_prefix + conv_prefix + "_weight",
        shape=(32, 4, 3, 3),
        dtype=flow.float32,
        initializer=weight_init,
        trainable=is_train,
    )
    conv1_bias = flow.get_variable(
        var_name_prefix + conv_prefix + "_bias",
        shape=(32,),
        dtype=flow.float32,
        initializer=bias_init,
        trainable=is_train,
    )

    conv_prefix = "_conv2"
    conv2_weight = flow.get_variable(
        var_name_prefix + conv_prefix + "_weight",
        shape=(32, 32, 3, 3),
        dtype=flow.float32,
        initializer=weight_init,
        trainable=is_train,
    )
    conv2_bias = flow.get_variable(
        var_name_prefix + conv_prefix + "_bias",
        shape=(32,),
        dtype=flow.float32,
        initializer=bias_init,
        trainable=is_train,
    )

    fc_prefix = "_fc1"
    fc1_weight = flow.get_variable(
        var_name_prefix + fc_prefix + "_weight",
        shape=(512, 32 * 16 * 16),
        dtype=flow.float32,
        initializer=weight_init,
        trainable=is_train,
    )
    fc1_bias = flow.get_variable(
        var_name_prefix + fc_prefix + "_bias",
        shape=(512,),
        dtype=flow.float32,
        initializer=bias_init,
        trainable=is_train,
    )

    fc_prefix = "_fc2"
    fc2_weight = flow.get_variable(
        var_name_prefix + fc_prefix + "_weight",
        shape=(2, 512),
        dtype=flow.float32,
        initializer=weight_init,
        trainable=is_train,
    )
    fc2_bias = flow.get_variable(
        var_name_prefix + fc_prefix + "_bias",
        shape=(2,),
        dtype=flow.float32,
        initializer=bias_init,
        trainable=is_train,
    )

    return (
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    )


BATCH_SIZE = 32


def createOfQNet(
    input_image: oft.Numpy.Placeholder((BATCH_SIZE, 4, 64, 64), dtype=flow.float32),
    var_name_prefix: str = "QNet",
    is_train: bool = True,
) -> oft.Numpy:

    (
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    ) = getQNetParams(var_name_prefix=var_name_prefix, is_train=is_train)

    (
        conv1_weight,
        conv1_bias,
        conv2_weight,
        conv2_bias,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
    ) = getQNetParams(var_name_prefix=var_name_prefix, is_train=is_train)

    conv1 = flow.nn.compat_conv2d(
        input_image, conv1_weight, strides=[1, 1], padding="same", data_format="NCHW"
    )
    conv1 = flow.nn.bias_add(conv1, conv1_bias, "NCHW")
    conv1 = flow.nn.relu(conv1)

    pool1 = flow.nn.max_pool2d(conv1, 2, 2, "VALID", "NCHW", name="pool1")

    conv2 = flow.nn.compat_conv2d(
        pool1, conv2_weight, strides=[1, 1], padding="same", data_format="NCHW"
    )
    conv2 = flow.nn.bias_add(conv2, conv2_bias, "NCHW")
    conv2 = flow.nn.relu(conv2)

    pool2 = flow.nn.max_pool2d(conv2, 2, 2, "VALID", "NCHW", name="pool2")

    # conv3.shape = (32, 32, 16, 16), after reshape become (32, 32 * 16 * 16)
    pool2_flatten = flow.reshape(pool2, (BATCH_SIZE, -1))
    fc1 = flow.matmul(a=pool2_flatten, b=fc1_weight, transpose_b=True)
    fc1 = flow.nn.bias_add(fc1, fc1_bias)
    fc1 = flow.nn.relu(fc1)

    fc2 = flow.matmul(a=fc1, b=fc2_weight, transpose_b=True)
    fc2 = flow.nn.bias_add(fc2, fc2_bias)

    return fc2


def get_train_config():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_logical_view(flow.scope.consistent_view())
    return func_config


def get_predict_config():
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    func_config.default_logical_view(flow.scope.consistent_view())
    return func_config


class DQN:
    def __init__(self, device_tag):
        self.device_tag_ = device_tag

    def test_parameters_copy(self):
        @flow.global_function("train", get_train_config())
        def trainQNet(
            input_image: oft.Numpy.Placeholder(
                (BATCH_SIZE, 4, 64, 64), dtype=flow.float32
            ),
            y_input: oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.float32),
            action_input: oft.Numpy.Placeholder((BATCH_SIZE, 2), dtype=flow.float32),
        ) -> oft.Numpy:
            with flow.scope.placement(self.device_tag_, "0:0-0"):
                out = createOfQNet(input_image, var_name_prefix="QNet", is_train=True)
                Q_Action = flow.math.reduce_sum(out * action_input, axis=1)
                cost = flow.math.reduce_mean(flow.math.square(y_input - Q_Action))
                learning_rate = 0.0002
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                    momentum=0,
                ).minimize(cost)
            return out

        @flow.global_function("predict", get_predict_config())
        def predictQNet(
            input_image: oft.Numpy.Placeholder(
                (BATCH_SIZE, 4, 64, 64), dtype=flow.float32
            )
        ) -> oft.Numpy:
            with flow.scope.placement(self.device_tag_, "0:0-0"):
                out = createOfQNet(input_image, var_name_prefix="QNetT", is_train=False)
                return out

        # copy QNet parameters to QNetT
        @flow.global_function("predict", get_predict_config())
        def copyQNetToQnetT():
            with flow.scope.placement(self.device_tag_, "0:0-0"):
                (
                    t_conv1_weight,
                    t_conv1_bias,
                    t_conv2_weight,
                    t_conv2_bias,
                    t_fc1_weight,
                    t_fc1_bias,
                    t_fc2_weight,
                    t_fc2_bias,
                ) = getQNetParams(var_name_prefix="QNet", is_train=True)
                (
                    p_conv1_weight,
                    p_conv1_bias,
                    p_conv2_weight,
                    p_conv2_bias,
                    p_fc1_weight,
                    p_fc1_bias,
                    p_fc2_weight,
                    p_fc2_bias,
                ) = getQNetParams(var_name_prefix="QNetT", is_train=False)

                flow.assign(p_conv1_weight, t_conv1_weight)
                flow.assign(p_conv1_bias, t_conv1_bias)
                flow.assign(p_conv2_weight, t_conv2_weight)
                flow.assign(p_conv2_bias, t_conv2_bias)
                flow.assign(p_fc1_weight, t_fc1_weight)
                flow.assign(p_fc1_bias, t_fc1_bias)
                flow.assign(p_fc2_weight, t_fc2_weight)
                flow.assign(p_fc2_bias, t_fc2_bias)

        check_point = flow.train.CheckPoint()
        check_point.init()

        input_image = np.ones((BATCH_SIZE, 4, 64, 64)).astype(np.float32)
        y_input = np.random.random_sample((BATCH_SIZE,)).astype(np.float32)
        action_input = np.random.random_sample((BATCH_SIZE, 2)).astype(np.float32)

        train_out = trainQNet(input_image, y_input, action_input)
        copyQNetToQnetT()

        train_out = trainQNet(input_image, y_input, action_input)
        predict_out = predictQNet(input_image)

        assert np.allclose(
            train_out, predict_out, rtol=1e-2, atol=1e-1
        ), "{}-{}".format(train_out.mean(), predict_out.mean())

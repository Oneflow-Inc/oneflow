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
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import test_global_storage
import oneflow.typing as tp

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(
    test_case, device_type, data_type, shape, axis, reverse, exclusive
):
    assert device_type in ["gpu", "cpu"]
    assert axis >= 0 and axis < len(shape)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    flow_type = type_name_to_flow_type[data_type]

    @flow.global_function(type="train", function_config=func_config)
    def CumsumJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            x = flow.cast(x, dtype=flow_type)
            out = flow.math.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse)
            loss = flow.cast(out, dtype=flow.float)
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
    of_out = CumsumJob().get()

    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"), dtype=tf.float32)
        tf_out = tf.math.cumsum(x, axis=axis, exclusive=exclusive, reverse=reverse)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    assert np.allclose(of_out.numpy(), tf_out.numpy(), atol=1e-03), np.max(
        np.abs(of_out.numpy() - tf_out.numpy())
    )
    assert np.allclose(test_global_storage.Get("x_diff"), tf_x_diff.numpy(), atol=1e-03)


def test_cumsum(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["data_type"] = ["double", "float32", "int64", "int32"]
    arg_dict["shape"] = [(5, 4, 3)]
    arg_dict["axis"] = [0, 1, 2]
    arg_dict["reverse"] = [True, False]
    arg_dict["exclusive"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(test_case, *arg)

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
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
from test_util import GenArgList
import oneflow.typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def _np_dtype_to_of_dtype(np_dtype):
    if np_dtype == np.float32:
        return flow.float
    else:
        raise NotImplementedError


def _of_clip_by_value(values, min, max, device_type="gpu", dynamic=False, grad_cb=None):
    data_type = _np_dtype_to_of_dtype(values.dtype)

    if callable(grad_cb):

        def clip(values_blob):
            with flow.scope.placement(device_type, "0:0"):
                x = flow.get_variable(
                    "values",
                    shape=values.shape,
                    dtype=data_type,
                    initializer=flow.constant_initializer(0),
                )
                x = flow.cast_to_current_logical_view(x)
                x = x + values_blob
                y = flow.clip_by_value(x, min, max)
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
                ).minimize(y)

            flow.watch_diff(x, grad_cb)
            return y

    else:

        def clip(values_blob):
            with flow.scope.placement(device_type, "0:0"):
                return flow.clip_by_value(values_blob, min, max, name="Clip")

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(data_type)
    if grad_cb is not None:
        func_config_type = "train"
    else:
        func_config_type = "predict"

    if dynamic:
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(type=func_config_type, function_config=func_config)
        def clip_fn(
            values_def: oft.ListNumpy.Placeholder(values.shape, dtype=data_type)
        ):
            return clip(values_def)

        return clip_fn([values]).get().numpy_list()[0]

    else:
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(type=func_config_type, function_config=func_config)
        def clip_fn(values_def: oft.Numpy.Placeholder(values.shape, dtype=data_type)):
            return clip(values_def)

        return clip_fn(values).get().numpy()


def _compare_with_tf(test_case, values, min, max, device_type, dynamic):
    with tf.GradientTape() as t:
        x = tf.Variable(values)
        y = tf.clip_by_value(x, min, max)
    dy = t.gradient(y, x)

    def compare_dy(dy_blob):
        test_case.assertTrue(
            np.array_equal(
                dy.numpy(), dy_blob.numpy_list()[0] if dynamic else dy_blob.numpy()
            )
        )

    of_y = _of_clip_by_value(
        values=values,
        min=min,
        max=max,
        device_type=device_type,
        dynamic=dynamic,
        grad_cb=compare_dy,
    )
    test_case.assertTrue(np.array_equal(y.numpy(), of_y))


@flow.unittest.skip_unless_1n1d()
class TestClipByValue(flow.unittest.TestCase):
    def test_clip_by_value(test_case):
        values = np.random.randint(low=-100, high=100, size=(8, 512, 4)).astype(
            np.float32
        )
        np_out = np.clip(values, -50, 50)

        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["dynamic"] = [True, False]
        for arg in GenArgList(arg_dict):
            of_out = _of_clip_by_value(values, -50, 50, *arg)
            test_case.assertTrue(np.array_equal(np_out, of_out))

    def test_clip_by_min(test_case):
        values = np.random.standard_normal((100, 30)).astype(np.float32)
        np_out = np.clip(values, a_min=0, a_max=None)
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["dynamic"] = [True, False]
        for arg in GenArgList(arg_dict):
            of_out = _of_clip_by_value(values, 0, None, *arg)
            test_case.assertTrue(np.array_equal(np_out, of_out))

    def test_clip_by_max(test_case):
        values = np.random.standard_normal((2, 64, 800, 1088)).astype(np.float32)
        np_out = np.clip(values, a_min=None, a_max=0.2)
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["dynamic"] = [True, False]
        for arg in GenArgList(arg_dict):
            of_out = _of_clip_by_value(values, None, 0.2, *arg)
            test_case.assertTrue(np.allclose(np_out, of_out))

    def test_clip_by_value_grad(test_case):
        values = np.random.standard_normal(1024).astype(np.float32)
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["dynamic"] = [True, False]
        for arg in GenArgList(arg_dict):
            _compare_with_tf(test_case, values, 0, 0.5, *arg)

    def test_clip_by_value_grad_case_1(test_case):
        values = np.random.standard_normal((128, 10, 27)).astype(np.float32)
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["dynamic"] = [True, False]
        for arg in GenArgList(arg_dict):
            _compare_with_tf(test_case, values, -0.2, 0.2, *arg)


if __name__ == "__main__":
    unittest.main()

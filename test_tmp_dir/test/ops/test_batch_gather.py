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
from tensorflow.python.ops import gen_math_ops
from test_util import GenArgList
import oneflow.typing as oft

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def _random_inputs(params_shape, indices_shape):
    params = np.random.rand(*params_shape).astype(np.float32)
    indices = np.random.randint(
        low=0,
        high=params_shape[len(indices_shape) - 1],
        size=indices_shape,
        dtype=np.int32,
    )
    return params, indices


def _make_gather_fn(
    params, indices, axis, batch_dims, device_type, mirrored, compare_fn
):
    flow.clear_default_session()
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    if mirrored:
        func_config.default_logical_view(flow.scope.mirrored_view())
    else:
        func_config.default_logical_view(flow.scope.consistent_view())

    def do_gather(x_blob, i_blob):
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "params",
                shape=params.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
            )
            x = x + x_blob
            y = flow.gather(x, i_blob, axis=axis, batch_dims=batch_dims)
            lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [1e-3])
            flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(y)
        flow.watch_diff(x, compare_fn)
        return y

    if mirrored:

        @flow.global_function(type="train", function_config=func_config)
        def gather_fn(
            params_def: oft.ListNumpy.Placeholder(params.shape, dtype=flow.float32),
            indices_def: oft.ListNumpy.Placeholder(indices.shape, dtype=flow.int32),
        ):
            return do_gather(params_def, indices_def)

    else:

        @flow.global_function(type="train", function_config=func_config)
        def gather_fn(
            params_def: oft.Numpy.Placeholder(params.shape, dtype=flow.float32),
            indices_def: oft.Numpy.Placeholder(indices.shape, dtype=flow.int32),
        ):
            return do_gather(params_def, indices_def)

    return gather_fn


def _compare_gather_with_tf(
    test_case,
    device_type,
    params_shape,
    indices_shape,
    axis,
    batch_dims,
    mirrored=False,
):
    params, indices = _random_inputs(params_shape, indices_shape)
    i = tf.constant(indices.astype(np.int32))
    with tf.GradientTape() as t:
        x = tf.Variable(params.astype(np.float32))
        y = tf.gather(x, i, axis=axis, batch_dims=axis)
    dy = t.gradient(y, x)
    if mirrored:

        def compare_dy(params_grad):
            test_case.assertTrue(
                np.allclose(dy, params_grad.numpy_list()[0], atol=1e-5, rtol=1e-5)
            )

    else:

        def compare_dy(params_grad):
            test_case.assertTrue(
                np.allclose(dy, params_grad.numpy(), atol=1e-5, rtol=1e-5)
            )

    gather_fn = _make_gather_fn(
        params, indices, axis, batch_dims, device_type, mirrored, compare_dy
    )

    if mirrored:
        of_y = gather_fn([params], [indices]).get().numpy_list()[0]
    else:
        of_y = gather_fn(params, indices).get().numpy()
    test_case.assertTrue(np.array_equal(y.numpy(), of_y))


@flow.unittest.skip_unless_1n1d()
class TestBatchGather(flow.unittest.TestCase):
    def test_batch_gather(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["params_shape"] = [(2, 8, 4)]
        arg_dict["indices_shape"] = [(2, 1)]
        arg_dict["axis"] = [1]
        arg_dict["batch_dims"] = [1]
        for arg in GenArgList(arg_dict):
            _compare_gather_with_tf(test_case, *arg)

    def test_batch_gather_case_1(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["params_shape"] = [(20, 10, 200)]
        arg_dict["indices_shape"] = [(20, 10)]
        arg_dict["axis"] = [1]
        arg_dict["batch_dims"] = [1]
        for arg in GenArgList(arg_dict):
            _compare_gather_with_tf(test_case, *arg)

    def test_batch_gather_case_2(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["params_shape"] = [(20, 80, 30, 5)]
        arg_dict["indices_shape"] = [(20, 40)]
        arg_dict["axis"] = [1]
        arg_dict["batch_dims"] = [1]
        arg_dict["mirrored"] = [True]
        for arg in GenArgList(arg_dict):
            _compare_gather_with_tf(test_case, *arg)

    def test_batch_gather_case_3(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["params_shape"] = [(20, 80, 30, 5)]
        arg_dict["indices_shape"] = [(20, 80, 20)]
        arg_dict["axis"] = [2]
        arg_dict["batch_dims"] = [2]
        arg_dict["mirrored"] = [True]
        for arg in GenArgList(arg_dict):
            _compare_gather_with_tf(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

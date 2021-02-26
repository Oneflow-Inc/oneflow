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
import oneflow_api
import tensorflow as tf
from test_util import GenArgList
import oneflow.typing as oft
import test_global_storage

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_reduce_any_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.int8)

    @flow.global_function(function_config=func_config)
    def ReduceAnyJob(x: oft.Numpy.Placeholder(input_shape, dtype=flow.int8)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_any(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.int8)
    # OneFlow
    of_out = ReduceAnyJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_any(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


def compare_reduce_prod_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(function_config=func_config)
    def ReduceProdJob(x: oft.Numpy.Placeholder(input_shape, dtype=flow.float32)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_prod(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceProdJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_prod(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


def compare_reduce_min_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def ReduceMinJob(x: oft.Numpy.Placeholder(input_shape, dtype=flow.float)):
        with flow.scope.placement(device_type, "0:0"):
            x += flow.get_variable(
                name="v1",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            loss = flow.math.reduce_min(x, axis=axis, keepdims=keepdims)
            loss = flow.identity(loss)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceMinJob(x).get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        tf_out = tf.math.reduce_min(x, axis=axis, keepdims=keepdims)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(
        test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )


def compare_reduce_all_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.int8)

    @flow.global_function(function_config=func_config)
    def ReduceAllJob(x: oft.Numpy.Placeholder(input_shape, dtype=flow.int8)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_all(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.int8)
    # OneFlow
    of_out = ReduceAllJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_all(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


def compare_reduce_sum_with_tensorflow(
    test_case, device_type, input_shape, axis, keepdims
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.int32)

    @flow.global_function(function_config=func_config)
    def ReduceSumJob(x: oft.Numpy.Placeholder(input_shape, dtype=flow.int32)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_sum(x, axis=axis, keepdims=keepdims)

    x = (np.random.rand(*input_shape) * 100).astype(np.int32)
    # OneFlow
    of_out = ReduceSumJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_sum(x, axis=axis, keepdims=keepdims)
    test_case.assertTrue(np.allclose(of_out.numpy(), tf_out.numpy()))


def compare_reduce_euclidean_norm_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(function_config=func_config)
    def ReduceEuclideanNormJob(x: oft.Numpy.Placeholder(input_shape, dtype=flow.float)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_euclidean_norm(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceEuclideanNormJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_euclidean_norm(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


def compare_reduce_logsumexp_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(function_config=func_config)
    def ReduceLogSumExpJob(x: oft.Numpy.Placeholder(input_shape, dtype=flow.float)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceLogSumExpJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


def compare_reduce_std_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(function_config=func_config)
    def ReduceStdJob(x: oft.Numpy.Placeholder(input_shape, dtype=flow.float)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_std(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceStdJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_std(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


def compare_reduce_variance_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(function_config=func_config)
    def ReduceVarianceJob(x: oft.Numpy.Placeholder(input_shape, dtype=flow.float)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_variance(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceVarianceJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


def compare_reduce_max_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def ReduceMaxJob(x: oft.Numpy.Placeholder(input_shape, dtype=flow.float)):
        with flow.scope.placement(device_type, "0:0"):
            x += flow.get_variable(
                name="v1",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            loss = flow.math.reduce_max(x, axis=axis, keepdims=keepdims)
            loss = flow.identity(loss)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceMaxJob(x).get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        tf_out = tf.math.reduce_max(x, axis=axis, keepdims=keepdims)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)
    assert np.allclose(
        test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )


@flow.unittest.skip_unless_1n2d()
class TestReduceOps(flow.unittest.TestCase):
    def test_reduce_any_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_any_with_tensorflow(*arg)

    def test_reduce_any_with_one_value_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1,)]
        arg_dict["axis"] = [None, [], [0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_any_with_tensorflow(*arg)

    def test_reduce_any_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_any_with_tensorflow(*arg)

    def test_reduce_any_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_any_with_tensorflow(*arg)

    def test_reduce_any_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_any_with_tensorflow(*arg)

    def test_reduce_any_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,), dtype=flow.int8)):
            y = flow.math.reduce_any(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.int8))

    def test_reduce_prod_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_prod_with_tensorflow(*arg)

    def test_reduce_prod_with_one_value_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1,)]
        arg_dict["axis"] = [None, [], [0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_prod_with_tensorflow(*arg)

    def test_reduce_prod_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_prod_with_tensorflow(*arg)

    def test_reduce_prod_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_prod_with_tensorflow(*arg)

    def test_reduce_prod_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_prod_with_tensorflow(*arg)

    def test_reduce_prod_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.reduce_prod(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.float32))

    def test_reduce_min_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_min_with_tensorflow(*arg)

    def test_reduce_min_with_one_value_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1,)]
        arg_dict["axis"] = [None, [], [0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_min_with_tensorflow(*arg)

    def test_reduce_min_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_min_with_tensorflow(*arg)

    def test_reduce_min_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_min_with_tensorflow(*arg)

    def test_reduce_min_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_min_with_tensorflow(*arg)

    def test_reduce_min_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.reduce_min(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.float32))

    def test_reduce_all_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_all_with_tensorflow(*arg)

    def test_reduce_all_with_one_value_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1,)]
        arg_dict["axis"] = [None, [], [0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_all_with_tensorflow(*arg)

    def test_reduce_all_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_all_with_tensorflow(*arg)

    def test_reduce_all_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_all_with_tensorflow(*arg)

    def test_reduce_all_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_all_with_tensorflow(*arg)

    def test_reduce_all_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,), dtype=flow.int8)):
            y = flow.math.reduce_all(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.int8))

    def test_reduce_sum_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_sum_with_tensorflow(test_case, *arg)

    def test_reduce_sum_with_one_value_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1,)]
        arg_dict["axis"] = [None, [], [0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_sum_with_tensorflow(test_case, *arg)

    def test_reduce_sum_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_sum_with_tensorflow(test_case, *arg)

    def test_reduce_sum_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_sum_with_tensorflow(test_case, *arg)

    def test_reduce_sum_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_sum_with_tensorflow(test_case, *arg)

    def test_reduce_sum_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.reduce_sum(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.float32))

    def test_reduce_euclidean_norm_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_euclidean_norm_with_tensorflow(*arg)

    def test_reduce_euclidean_norm_with_one_value_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1,)]
        arg_dict["axis"] = [None, [], [0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_euclidean_norm_with_tensorflow(*arg)

    def test_reduce_euclidean_norm_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_euclidean_norm_with_tensorflow(*arg)

    def test_reduce_euclidean_norm_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_euclidean_norm_with_tensorflow(*arg)

    def test_reduce_euclidean_norm_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_euclidean_norm_with_tensorflow(*arg)

    def test_reduce_euclidean_norm_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.reduce_euclidean_norm(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.float32))

    def test_reduce_logsumexp_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_logsumexp_with_tensorflow(*arg)

    def test_reduce_logsumexp_with_one_value_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1,)]
        arg_dict["axis"] = [None, [], [0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_logsumexp_with_tensorflow(*arg)

    def test_reduce_logsumexp_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_logsumexp_with_tensorflow(*arg)

    def test_reduce_logsumexp_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_logsumexp_with_tensorflow(*arg)

    def test_reduce_logsumexp_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_logsumexp_with_tensorflow(*arg)

    def test_reduce_logsumexp_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.reduce_logsumexp(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.float32))

    def test_reduce_std_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_std_with_tensorflow(*arg)

    def test_reduce_std_with_one_value_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1,)]
        arg_dict["axis"] = [None, [], [0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_std_with_tensorflow(*arg)

    def test_reduce_std_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_std_with_tensorflow(*arg)

    def test_reduce_std_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_std_with_tensorflow(*arg)

    def test_reduce_std_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_std_with_tensorflow(*arg)

    def test_reduce_std_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.reduce_std(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.float32))

    def test_reduce_variance_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_variance_with_tensorflow(*arg)

    def test_reduce_variance_with_one_value_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1,)]
        arg_dict["axis"] = [None, [], [0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_variance_with_tensorflow(*arg)

    def test_reduce_variance_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_variance_with_tensorflow(*arg)

    def test_reduce_variance_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_variance_with_tensorflow(*arg)

    def test_reduce_variance_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_variance_with_tensorflow(*arg)

    def test_reduce_variance_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.reduce_variance(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.float32))

    def test_reduce_max_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 64, 64)]
        arg_dict["axis"] = [None, [], [1], [0, 2]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_max_with_tensorflow(*arg)

    def test_reduce_max_with_one_value_func(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1,)]
        arg_dict["axis"] = [None, [], [0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_max_with_tensorflow(*arg)

    def test_reduce_max_col_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_max_with_tensorflow(*arg)

    def test_reduce_max_row_reduce(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(25, 1024 * 1024)]
        arg_dict["axis"] = [[1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_max_with_tensorflow(*arg)

    def test_reduce_max_scalar(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(1024 * 64, 25)]
        arg_dict["axis"] = [[0, 1]]
        arg_dict["keepdims"] = [True, False]
        for arg in GenArgList(arg_dict):
            compare_reduce_max_with_tensorflow(*arg)

    def test_reduce_max_split_axis_reduced(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.reduce_max(x)
            test_case.assertTrue(y.split_axis == flow.INVALID_SPLIT_AXIS)

        Foo(np.ndarray((10,), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

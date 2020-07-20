import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.core.common.data_type_pb2 as data_type_util
import tensorflow as tf
from test_util import GenArgList


def compare_reduce_any_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.int8)

    @flow.global_function(func_config)
    def ReduceAnyJob(x=flow.FixedTensorDef(input_shape, dtype=data_type_util.kInt8)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_any(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.int8)
    # OneFlow
    of_out = ReduceAnyJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_any(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


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


def test_reduce_any_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,), dtype=data_type_util.kInt8)):
        y = flow.math.reduce_any(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.int8))


def compare_reduce_prod_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(func_config)
    def ReduceProdJob(x=flow.FixedTensorDef(input_shape, dtype=data_type_util.kFloat)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_prod(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceProdJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_prod(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


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


def test_reduce_prod_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,))):
        y = flow.math.reduce_prod(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.float32))


def compare_reduce_min_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(func_config)
    def ReduceMinJob(x=flow.FixedTensorDef(input_shape, dtype=data_type_util.kFloat)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_min(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceMinJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_min(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


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


def test_reduce_min_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,))):
        y = flow.math.reduce_min(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.float32))


def compare_reduce_all_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.int8)

    @flow.global_function(func_config)
    def ReduceAllJob(x=flow.FixedTensorDef(input_shape, dtype=data_type_util.kInt8)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_all(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.int8)
    # OneFlow
    of_out = ReduceAllJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_all(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


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


def test_reduce_all_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,), dtype=data_type_util.kInt8)):
        y = flow.math.reduce_all(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.int8))


def compare_reduce_sum_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(func_config)
    def ReduceSumJob(x=flow.FixedTensorDef(input_shape, dtype=data_type_util.kFloat)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_sum(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceSumJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_sum(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


def test_reduce_sum_func(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(64, 64, 64)]
    arg_dict["axis"] = [None, [], [1], [0, 2]]
    arg_dict["keepdims"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_reduce_sum_with_tensorflow(*arg)


def test_reduce_sum_with_one_value_func(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(1,)]
    arg_dict["axis"] = [None, [], [0]]
    arg_dict["keepdims"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_reduce_sum_with_tensorflow(*arg)


def test_reduce_sum_col_reduce(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(1024 * 64, 25)]
    arg_dict["axis"] = [[0]]
    arg_dict["keepdims"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_reduce_sum_with_tensorflow(*arg)


def test_reduce_sum_row_reduce(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(25, 1024 * 1024)]
    arg_dict["axis"] = [[1]]
    arg_dict["keepdims"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_reduce_sum_with_tensorflow(*arg)


def test_reduce_sum_scalar(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(1024 * 64, 25)]
    arg_dict["axis"] = [[0, 1]]
    arg_dict["keepdims"] = [True, False]
    for arg in GenArgList(arg_dict):
        compare_reduce_sum_with_tensorflow(*arg)


def test_reduce_sum_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,))):
        y = flow.math.reduce_sum(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.float32))


def compare_reduce_euclidean_norm_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(func_config)
    def ReduceEuclideanNormJob(
        x=flow.FixedTensorDef(input_shape, dtype=data_type_util.kFloat)
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_euclidean_norm(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceEuclideanNormJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_euclidean_norm(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


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


def test_reduce_euclidean_norm_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,))):
        y = flow.math.reduce_euclidean_norm(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.float32))


def compare_reduce_logsumexp_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(func_config)
    def ReduceLogSumExpJob(
        x=flow.FixedTensorDef(input_shape, dtype=data_type_util.kFloat)
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceLogSumExpJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_logsumexp(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


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


def test_reduce_logsumexp_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,))):
        y = flow.math.reduce_logsumexp(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.float32))


def compare_reduce_std_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(func_config)
    def ReduceStdJob(x=flow.FixedTensorDef(input_shape, dtype=data_type_util.kFloat)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_std(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceStdJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_std(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


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


def test_reduce_std_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,))):
        y = flow.math.reduce_std(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.float32))


def compare_reduce_variance_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(func_config)
    def ReduceVarianceJob(
        x=flow.FixedTensorDef(input_shape, dtype=data_type_util.kFloat)
    ):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_variance(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceVarianceJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


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


def test_reduce_variance_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,))):
        y = flow.math.reduce_variance(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.float32))


def compare_reduce_max_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function(func_config)
    def ReduceMaxJob(x=flow.FixedTensorDef(input_shape, dtype=data_type_util.kFloat)):
        with flow.scope.placement(device_type, "0:0"):
            return flow.math.reduce_max(x, axis=axis, keepdims=keepdims)

    x = np.random.rand(*input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceMaxJob(x).get()
    # TensorFlow
    tf_out = tf.math.reduce_max(x, axis=axis, keepdims=keepdims)
    assert np.allclose(of_out.numpy(), tf_out.numpy(), rtol=rtol, atol=atol)


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


def test_reduce_max_batch_axis_reduced(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,))):
        y = flow.math.reduce_max(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.float32))

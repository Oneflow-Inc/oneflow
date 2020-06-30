import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
import test_global_storage
from test_util import GenArgList


def compare_reduce_sum_with_tensorflow(
    device_type, input_shape, axis, keepdims, rtol=1e-5, atol=1e-5
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(func_config)
    def ReduceSumJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "in",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=2, maxval=5),
                trainable=True,
            )
            loss = flow.math.reduce_sum(x, axis=axis, keepdims=keepdims)
            flow.losses.add_loss(loss)
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))
            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = ReduceSumJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        tf_out = tf.math.reduce_sum(x, axis=axis, keepdims=keepdims)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )


def test_reduce_sum_func(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(64, 64, 64)]
    arg_dict["axis"] = [None, [], [1], [0, 2]]
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
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.global_function(func_config)
    def Foo(x=flow.FixedTensorDef((10,))):
        y = flow.math.reduce_sum(x)
        test_case.assertTrue(y.split_axis is None)
        test_case.assertTrue(y.batch_axis is None)

    Foo(np.ndarray((10,), dtype=np.float32))

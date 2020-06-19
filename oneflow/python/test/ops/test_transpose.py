import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
import test_global_storage
from test_util import GenArgList


def compare_with_tensorflow(device_type, input_shape, perm):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(func_config)
    def TransposeJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "input",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=2, maxval=5),
                trainable=True,
            )

            loss = flow.transpose(x, perm)
            flow.losses.add_loss(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = TransposeJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        tf_out = tf.transpose(x, perm)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )


def test_transpose(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(10, 11, 12, 13)]
    arg_dict["perm"] = [(2, 0, 1, 3), (1, 0, 2, 3), (3, 2, 1, 0), (3, 1, 2, 0)]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)


def test_transpose2(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(10, 11, 12)]
    arg_dict["perm"] = [(2, 0, 1), (1, 0, 2), (2, 1, 0), (1, 2, 0)]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)


def test_transpose3(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(10, 11)]
    arg_dict["perm"] = [(1, 0), (0, 1)]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

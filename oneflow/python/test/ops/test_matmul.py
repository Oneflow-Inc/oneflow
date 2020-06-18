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


def compare_with_tensorflow(device_type, a_shape, b_shape, transpose_a, transpose_b):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.function(func_config)
    def MatmulJob():
        with flow.device_prior_placement(device_type, "0:0"):
            a = flow.get_variable(
                "a",
                shape=a_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            b = flow.get_variable(
                "b",
                shape=b_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = flow.matmul(a, b, transpose_a, transpose_b)
            flow.losses.add_loss(loss)

            flow.watch(a, test_global_storage.Setter("a"))
            flow.watch_diff(a, test_global_storage.Setter("a_diff"))
            flow.watch(b, test_global_storage.Setter("b"))
            flow.watch_diff(b, test_global_storage.Setter("b_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = MatmulJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        a = tf.Variable(test_global_storage.Get("a"))
        b = tf.Variable(test_global_storage.Get("b"))
        tf_out = tf.matmul(a, b, transpose_a, transpose_b)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_a_diff = tape.gradient(tf_out, a, loss_diff)
    tf_b_diff = tape.gradient(tf_out, b, loss_diff)

    assert np.allclose(of_out.ndarray(), tf_out.numpy(), atol=1e-03), np.max(
        np.abs(of_out.ndarray() - tf_out.numpy())
    )
    assert np.allclose(test_global_storage.Get("a_diff"), tf_a_diff.numpy(), atol=1e-03)
    assert np.allclose(test_global_storage.Get("b_diff"), tf_b_diff.numpy(), atol=1e-03)


def filter_args(arg_list):
    def trans_shape(shape):
        tmp_shape = shape[:-2]
        tmp_shape += (shape[-1], shape[-2])
        return tmp_shape

    ret = []
    for arg in arg_list:
        a_shape = arg[1]
        b_shape = arg[2]
        if arg[3]:  # transpose_a
            a_shape = trans_shape(a_shape)
        if arg[4]:  # transpose_b
            b_shape = trans_shape(b_shape)
        if a_shape[-1] == b_shape[-2]:
            ret.append(tuple(arg))
    return ret


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["a_shape"] = [(512, 256), (256, 512)]
    arg_dict["b_shape"] = [(256, 1024), (1024, 256)]
    arg_dict["transpose_a"] = [True, False]
    arg_dict["transpose_b"] = [True, False]
    matmul_args = filter_args(GenArgList(arg_dict))

    arg_dict.clear()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["a_shape"] = [(10, 10, 64, 32), (10, 10, 32, 64)]
    arg_dict["b_shape"] = [(10, 10, 32, 128), (10, 10, 128, 32)]
    arg_dict["transpose_a"] = [True, False]
    arg_dict["transpose_b"] = [True, False]
    batch_matmul_args = filter_args(GenArgList(arg_dict))

    return matmul_args + batch_matmul_args


def test_matmul(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)

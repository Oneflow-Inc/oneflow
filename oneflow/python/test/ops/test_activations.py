import numpy as np
import math
import os
import oneflow as flow
import tensorflow as tf
import tensorflow_addons as tfa
from collections import OrderedDict 

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save


def compare_with_tensorflow(device_type, activation_type, shape):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    of_activation_map = {
        "relu": flow.keras.activations.relu,
        "sigmoid": flow.keras.activations.sigmoid,
        "tanh": flow.keras.activations.tanh,
        "gelu": flow.keras.activations.gelu,
    }
    tf_activation_map = {
        "relu": tf.nn.relu,
        "sigmoid": tf.math.sigmoid,
        "tanh": tf.math.tanh,
        "gelu": tfa.activations.gelu,
    }

    @flow.function(func_config)
    def ActivationJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = of_activation_map[activation_type](x)
            flow.losses.add_loss(loss)

            flow.watch(x, Save("x"))
            flow.watch_diff(x, Save("x_diff"))
            flow.watch(loss, Save("loss"))
            flow.watch_diff(loss, Save("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = ActivationJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(np.load(os.path.join(GetSavePath(), "x.npy")))
        tf_out = tf_activation_map[activation_type](x)
    loss_diff = np.load(os.path.join(GetSavePath(), "loss_diff.npy"))
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    rtol = 1e-3 if activation_type is "gelu" else 1e-5
    atol = 1e-3 if activation_type is "gelu" else 1e-5
    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol, atol)
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "x_diff.npy")), tf_x_diff.numpy(), rtol, atol
    )

def test_activations(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["activation_type"] = ["relu", "sigmoid", "tanh", "gelu"]
    arg_dict["shape"] = [(1024, 1024)]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

import numpy as np
import math
import os
import oneflow as flow
import tensorflow as tf
import torch

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save


def compare_with_tensorflow(activation_type, shape, device_type):
    flow.clear_default_session()
    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

    of_activation_map = {
        "relu": flow.keras.activations.relu,
        "sigmoid": flow.keras.activations.sigmoid,
        "tanh": flow.keras.activations.tanh,
    }
    tf_activation_map = {"relu": tf.nn.relu, "sigmoid": tf.math.sigmoid, "tanh": tf.math.tanh}

    @flow.function
    def ActivationJob():
        flow.config.train.primary_lr(1e-4)
        flow.config.train.model_update_conf(dict(naive_conf={}))
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

    assert np.allclose(of_out, tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "x_diff.npy")),
        tf_x_diff.numpy(),
        rtol=1e-05,
        atol=1e-05,
    )


def test_activations(test_case):
    for arg in GenArgList([["relu", "sigmoid", "tanh"], [(1024, 1024)], ["gpu"]]):
        compare_with_tensorflow(*arg)


# TODO: move gelu test to test_activations
def _test_gelu(device_type):
    flow.clear_default_session()
    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

    @flow.function
    def GeluJob(x=flow.input_blob_def((10,))):
        with flow.device_prior_placement(device_type, "0:0"):
            return flow.keras.activations.gelu(x)

    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    ratios = [-2, -1, 0, 1, 2]
    ones = np.ones((10,), dtype=np.float32)
    for r in ratios:
        x = ones * r
        of_out = GeluJob(x).get()
        torch_out = gelu(torch.tensor(x)).numpy()
        assert np.allclose(of_out, torch_out, rtol=1e-3, atol=1e-4)


def test_gelu(test_case):
    _test_gelu("gpu")
    _test_gelu("cpu")

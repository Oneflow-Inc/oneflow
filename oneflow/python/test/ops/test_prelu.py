import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict 

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def compare_with_tensorflow(device_type, x_shape, data_format, channel_shared):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.function(func_config)
    def PreluJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=0, maxval=100),
                trainable=True,
            )
            loss = flow.layers.PRelu(
                x, data_format=data_format, channel_shared=channel_shared, alpha_initializer = flow.constant_initializer(0.25), name="prelu")
            if channel_shared:
                alpha_shape = (1,)
            else:
                if data_format == "NCHW":
                    alpha_shape = (x_shape[1],)
                elif data_format == "NHWC":
                    alpha_shape = (x_shape[3],)
            alpha = flow.get_variable(
                "prelu-alpha",
                shape=alpha_shape,
                dtype=flow.float,
                initializer=flow.constant_initializer(0.25),
                )
            loss = flow.math.sqrt(loss)
            flow.losses.add_loss(loss)

            flow.watch(x, Save("x"))
            flow.watch_diff(x, Save("x_diff"))
            flow.watch(alpha, Save("alpha"))
            flow.watch_diff(alpha, Save("alpha_diff"))
            flow.watch(loss, Save("loss"))
            flow.watch_diff(loss, Save("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = PreluJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(np.load(os.path.join(GetSavePath(), "x.npy")))
        if channel_shared:
            alpha_shape = (1,)
        else:
            if data_format == "NCHW":
                alpha_shape = (1, x_shape[1], 1, 1)
            elif data_format == "NHWC":
                alpha_shape = (1, 1, 1, x_shape[3])
        alphas = tf.Variable(np.load(os.path.join(GetSavePath(), "alpha.npy")).reshape(alpha_shape))
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        tf_out = pos + neg
        tf_out2 = tf.math.sqrt(tf_out)

    loss_diff = np.load(os.path.join(GetSavePath(), "loss_diff.npy"))
    tf_x_diff = tape.gradient(tf_out2, x, loss_diff)
    tf_alpha_diff = tape.gradient(tf_out2, alphas, loss_diff)

    assert np.allclose(of_out.ndarray(), tf_out2.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "x_diff.npy")), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "alpha_diff.npy")), tf_alpha_diff.numpy(), rtol=1e-5, atol=1e-5
    )


def test_prelu(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["x_shape"] = [(10, 32, 20, 20)]
    arg_dict["data_format"] = ["NCHW", "NHWC"]
    arg_dict["channel_shared"] = [False]

    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

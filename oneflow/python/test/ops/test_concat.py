import os
import numpy as np
import tensorflow as tf
import oneflow as flow

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save


def compare_with_tensorflow(device_type, x_shape, y_shape, axis):
    flow.clear_default_session()
    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

    @flow.function
    def ConcatJob():
        flow.config.train.primary_lr(1e-4)
        flow.config.train.model_update_conf(dict(naive_conf={}))
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            y = flow.get_variable(
                "y",
                shape=y_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = flow.concat([x, y], axis)
            flow.losses.add_loss(loss)

            flow.watch(x, Save("x"))
            flow.watch_diff(x, Save("x_diff"))
            flow.watch(y, Save("y"))
            flow.watch_diff(y, Save("y_diff"))
            flow.watch(loss, Save("loss"))
            flow.watch_diff(loss, Save("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = ConcatJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(np.load(os.path.join(GetSavePath(), "x.npy")))
        y = tf.Variable(np.load(os.path.join(GetSavePath(), "y.npy")))
        tf_out = tf.concat([x, y], axis)
    loss_diff = np.load(os.path.join(GetSavePath(), "loss_diff.npy"))
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
    tf_y_diff = tape.gradient(tf_out, y, loss_diff)

    assert np.allclose(of_out, tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "x_diff.npy")), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "y_diff.npy")), tf_y_diff.numpy(), rtol=1e-5, atol=1e-5
    )


def test_concat(test_case):
    for arg in GenArgList([["gpu"], [(10, 20, 30)], [(10, 20, 30)], [0, 1, 2]]):
        compare_with_tensorflow(*arg)

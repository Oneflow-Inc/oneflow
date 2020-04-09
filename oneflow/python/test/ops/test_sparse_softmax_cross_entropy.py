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


def compare_with_tensorflow(device_type, num_classes, batch_size):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))


    @flow.function(func_config)
    def SparseSoftmaxCrossEntropyWithLogitsJob(
            labels=flow.FixedTensorDef((batch_size, ), dtype=flow.int32)
        ):
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=(batch_size, num_classes),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            prediction = flow.nn.softmax(logits=x)
            loss = flow.nn.sparse_cross_entropy(labels=labels, prediction=prediction)
            #loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=x)
            loss = flow.identity(loss)
            flow.losses.add_loss(loss)

            flow.watch(x, Save("x"))
            flow.watch_diff(x, Save("x_diff"))
            flow.watch(loss, Save("loss"))
            flow.watch_diff(loss, Save("loss_diff"))
        return loss

    # fake labels
    labels = np.random.randint(0, num_classes, size=(batch_size, )).astype(np.int32)

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = SparseSoftmaxCrossEntropyWithLogitsJob(labels).get()

    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(np.load(os.path.join(GetSavePath(), "x.npy")))
        tf_out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, x)
    loss_diff = np.load(os.path.join(GetSavePath(), "loss_diff.npy"))
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "x_diff.npy")), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )
    flow.clear_default_session()


def test_sparse_softmax_cross_entropy_with_logits(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["num_classes"] = [1000]
    arg_dict["batch_size"] = [64]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

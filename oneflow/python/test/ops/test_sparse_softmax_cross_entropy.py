import os
import numpy as np
import tensorflow as tf
import oneflow as flow
from collections import OrderedDict

from test_util import GenArgList
import test_global_storage
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, data_type, label_type, num_classes, batch_size):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))


    @flow.function(func_config)
    def SparseSoftmaxCrossEntropyWithLogitsJob(
            labels=flow.FixedTensorDef((batch_size, ), dtype=type_name_to_flow_type[label_type])
        ):
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=(batch_size, num_classes),
                dtype=type_name_to_flow_type[data_type],
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=x)
            loss = flow.identity(loss)
            flow.losses.add_loss(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))
        return loss

    # fake labels
    labels = np.random.randint(0, num_classes, size=(batch_size, )).astype(type_name_to_np_type[label_type])

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = SparseSoftmaxCrossEntropyWithLogitsJob(labels).get()

    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        tf_out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, x)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )
    flow.clear_default_session()


def test_sparse_softmax_cross_entropy_with_logits(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["data_type"] = ["float32", "double"]
    arg_dict["label_type"] = ["int32", "int64"]
    arg_dict["num_classes"] = [1000]
    arg_dict["batch_size"] = [64]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

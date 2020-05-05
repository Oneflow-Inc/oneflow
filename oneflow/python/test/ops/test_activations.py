import numpy as np
import math
import os
import oneflow as flow
import tensorflow as tf
#import tensorflow_addons as tfa
from collections import OrderedDict 

from test_util import GenArgList
import test_global_storage

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def compare_with_tensorflow(device_type, activation_type, shape, data_type):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    flow.config.enable_debug_mode(True);
    func_config = flow.FunctionConfig()
    if data_type == flow.float16:
        func_config.enable_auto_mixed_precision(True)
        data_type = flow.float

    func_config.default_data_type(data_type)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    of_activation_map = {
        #"relu": flow.keras.activations.relu,
        "relu": flow.nn.relu,
        "sigmoid": flow.math.sigmoid,
        #"tanh": flow.keras.activations.tanh,
        "tanh": flow.math.tanh,
#        "gelu": flow.keras.activations.gelu,
    }
    tf_activation_map = {
        "relu": tf.nn.relu,
        "sigmoid": tf.math.sigmoid,
        "tanh": tf.math.tanh,
#        "gelu": tfa.activations.gelu,
    }

    @flow.function(func_config)
    def ActivationJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=shape,
                dtype=data_type,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = of_activation_map[activation_type](x)
            flow.losses.add_loss(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = ActivationJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        tf_out = tf_activation_map[activation_type](x)
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    rtol = 1e-3 if activation_type is "gelu" else 1e-5
    atol = 1e-3 if activation_type is "gelu" else 1e-5
    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol, atol)
    assert np.allclose(
        test_global_storage.Get("x_diff"), tf_x_diff.numpy(), rtol, atol
    )

def test_activations(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
#    arg_dict["activation_type"] = ["relu", "sigmoid", "tanh", "gelu"]
    arg_dict["activation_type"] = ["relu", "sigmoid", "tanh"]
    arg_dict["shape"] = [(1024, 1024)]
    arg_dict["data_type"] = [flow.float, flow.double]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

    for act_type in arg_dict["activation_type"]:
        compare_with_tensorflow('gpu', act_type, (1024, 1024), flow.float16)

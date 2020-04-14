import os
import numpy as np
import itertools
from collections import OrderedDict
from collections.abc import Iterable

import oneflow as flow
import tensorflow as tf


def GenCartesianProduct(sets):
    assert isinstance(sets, Iterable)
    for set in sets:
        assert isinstance(set, Iterable)
    return itertools.product(*sets)


def GenArgList(arg_dict):
    assert isinstance(arg_dict, OrderedDict)
    sets = [arg_set for _, arg_set in arg_dict.items()]
    return GenCartesianProduct(sets)


def GenArgDict(arg_dict):
    return [dict(zip(arg_dict.keys(), x)) for x in GenArgList(arg_dict)]


def GetSavePath():
    return "./log/op_unit_test/"


# Save func for flow.watch
def Save(name):
    path = GetSavePath()
    if not os.path.isdir(path):
        os.makedirs(path)

    def _save(x):
        np.save(os.path.join(path, name), x.ndarray())

    return _save


class Args:
    def __init__(self, flow_args, tf_args=None):
        super().__init__()
        if tf_args is None:
            tf_args = flow_args
        self.flow_args = flow_args
        self.tf_args = tf_args


def compare_with_tensorflow(param_dict):
    # necessary params
    device_type = param_dict['device_type']
    flow_op = param_dict['flow_op']
    tf_op = param_dict['tf_op']
    input_shape = param_dict['input_shape']
    # optional params
    op_args = param_dict.get('op_args')
    flow_initializer = param_dict.get('flow_initializer')

    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))
    if flow_initializer is None:
        flow_initializer = flow.random_uniform_initializer(minval=-10, maxval=10)
    if op_args is None:
        flow_args, tf_args = [], []
    else:
        flow_args, tf_args = op_args.flow_args, op_args.tf_args

    @flow.function(func_config)
    def FlowJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow_initializer,
                trainable=True,
            )
            loss = flow_op(x, *flow_args)
            flow.losses.add_loss(loss)

            flow.watch(x, Save("x"))
            flow.watch_diff(x, Save("x_diff"))
            flow.watch(loss, Save("loss"))
            flow.watch_diff(loss, Save("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = FlowJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(np.load(os.path.join(GetSavePath(), "x.npy")))
        tf_out = tf_op(x, *tf_args)
    loss_diff = np.load(os.path.join(GetSavePath(), "loss_diff.npy"))
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "x_diff.npy")), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )

type_name_to_flow_type = {
    "float16": flow.float16,
    "float32": flow.float32,
    "double": flow.double,
    "int8": flow.int8,
    "int32": flow.int32,
    "int64": flow.int64,
    "char": flow.char,
    "uint8": flow.uint8,
}

type_name_to_np_type = {
    "float16": np.float16,
    "float32": np.float32,
    "double": np.float64,
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
    "char": np.byte,
    "uint8": np.uint8,
}

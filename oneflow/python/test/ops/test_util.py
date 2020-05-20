import os
import numpy as np
import itertools
from collections import OrderedDict
from collections.abc import Iterable

import oneflow as flow
import tensorflow as tf
import test_global_storage

def GenCartesianProduct(sets):
    assert isinstance(sets, Iterable)
    for set in sets:
        assert isinstance(set, Iterable)
    return itertools.product(*sets)


def GenArgList(arg_dict):
    assert isinstance(arg_dict, OrderedDict)
    assert all([isinstance(x, list) for x in arg_dict.values()])
    sets = [arg_set for _, arg_set in arg_dict.items()]
    return GenCartesianProduct(sets)


def GenArgDict(arg_dict):
    return [dict(zip(arg_dict.keys(), x)) for x in GenArgList(arg_dict)]


class Args:
    def __init__(self, flow_args, tf_args=None):
        super().__init__()
        if tf_args is None:
            tf_args = flow_args
        self.flow_args = flow_args
        self.tf_args = tf_args


def RunOneflowOp(device_type, flow_op, x, flow_args):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(0)
    func_config.train.model_update_conf(dict(naive_conf={}))
    @flow.function(func_config)
    def FlowJob(x=flow.FixedTensorDef(x.shape)):
        with flow.device_prior_placement(device_type, "0:0"):
            x += flow.get_variable(name='v1', shape=(1,),
                                   dtype=flow.float, initializer=flow.zeros_initializer())
            loss = flow_op(x, *flow_args)
            flow.losses.add_loss(loss)

            flow.watch_diff(x, test_global_storage.Setter("x_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    y = FlowJob(x).get().ndarray()
    x_diff = test_global_storage.Get("x_diff")
    return y, x_diff


def RunTensorFlowOp(tf_op, x, tf_args):
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(x)
        y = tf_op(x, *tf_args)
    x_diff = tape.gradient(y, x)
    return y.numpy(), x_diff.numpy()


def CompareOpWithTensorFlow(device_type, flow_op, tf_op, input_shape,
                            op_args=None, input_minval=-10, input_maxval=10, y_rtol=1e-5,
                            y_atol=1e-5, x_diff_rtol=1e-5, x_diff_atol=1e-5):
    assert device_type in ["gpu", "cpu"]
    if op_args is None:
        flow_args, tf_args = [], []
    else:
        flow_args, tf_args = op_args.flow_args, op_args.tf_args

    x = np.random.uniform(low=input_minval, high=input_maxval,
                          size=input_shape).astype(np.float32)
    of_y, of_x_diff, = RunOneflowOp(device_type, flow_op, x, flow_args)
    tf_y, tf_x_diff = RunTensorFlowOp(tf_op, x, tf_args)

    assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)
    assert np.allclose(
        of_x_diff, tf_x_diff, rtol=x_diff_rtol, atol=x_diff_atol
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

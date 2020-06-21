import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import tensorflow as tf
import test_global_storage
from test_util import Args, GenArgDict

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def RunOneflowBiasAdd(device_type, value, bias, flow_args):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(0)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.global_function(func_config)
    def FlowJob(
        value=flow.FixedTensorDef(value.shape), bias=flow.FixedTensorDef(bias.shape)
    ):
        with flow.device_prior_placement(device_type, "0:0"):
            value += flow.get_variable(
                name="v1",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            bias += flow.get_variable(
                name="v2",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            loss = flow.nn.bias_add(value, bias, *flow_args)
            flow.losses.add_loss(loss)

            flow.watch_diff(value, test_global_storage.Setter("value_diff"))
            flow.watch_diff(bias, test_global_storage.Setter("bias_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    y = FlowJob(value, bias).get().ndarray()
    value_diff = test_global_storage.Get("value_diff")
    bias_diff = test_global_storage.Get("bias_diff")
    return y, value_diff, bias_diff


def RunTensorFlowBiasAdd(value, bias, tf_args):
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        value, bias = tf.Variable(value), tf.Variable(bias)
        y = tf.nn.bias_add(value, bias, *tf_args)
    value_diff = tape.gradient(y, value).numpy()
    bias_diff = tape.gradient(y, bias).numpy()
    return y.numpy(), value_diff, bias_diff


def CompareBiasAddWithTensorFlow(
    device_type,
    input_shapes,
    op_args=None,
    input_minval=-10,
    input_maxval=10,
    y_rtol=1e-5,
    y_atol=1e-5,
    x_diff_rtol=1e-5,
    x_diff_atol=1e-5,
):
    assert device_type in ["gpu", "cpu"]
    if op_args is None:
        flow_args, tf_args = [], []
    else:
        flow_args, tf_args = op_args.flow_args, op_args.tf_args

    x = [
        np.random.uniform(low=input_minval, high=input_maxval, size=input_shape).astype(
            np.float32
        )
        for input_shape in input_shapes
    ]
    of_y, of_x_diff1, of_x_diff2 = RunOneflowBiasAdd(device_type, *x, flow_args)
    tf_y, tf_x_diff1, tf_x_diff2 = RunTensorFlowBiasAdd(*x, tf_args)

    assert np.allclose(of_y, tf_y, rtol=y_rtol, atol=y_atol)
    assert np.allclose(of_x_diff1, tf_x_diff1, rtol=x_diff_rtol, atol=x_diff_atol)
    assert np.allclose(of_x_diff2, tf_x_diff2, rtol=x_diff_rtol, atol=x_diff_atol)


def test_bias_add_nchw(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["input_shapes"] = [((1, 20, 1, 11), (20,))]
    arg_dict["op_args"] = [Args(["NCHW"])]
    for arg in GenArgDict(arg_dict):
        CompareBiasAddWithTensorFlow(**arg)


def test_bias_add_nhwc(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["cpu", "gpu"]
    arg_dict["input_shapes"] = [((30, 20, 5, 10), (10,)), ((2, 5, 7, 8), (8,))]
    arg_dict["op_args"] = [Args(["NHWC"])]
    for arg in GenArgDict(arg_dict):
        CompareBiasAddWithTensorFlow(**arg)

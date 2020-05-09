import os
import oneflow as flow
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from test_util import GenArgList

import test_global_storage

oneflow2np_dtype_dict = {
    flow.int8: np.int8,
    flow.uint8: np.uint8,
    flow.int32: np.int32, 
    flow.int64: np.int64,
    flow.float: np.float32, 
    flow.double: np.float64, 
}

def cast_forward_compare_with_tensorflow(test_cast, device_type, input_shape, dtype, switch_dtype):
    assert device_type in ["gpu", "cpu"]    
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float) 
    @flow.function(func_config)
    def cast_forward(input_def=flow.FixedTensorDef(shape=input_shape, dtype=dtype)):
        with flow.fixed_placement(device_type, "0:0"):
            return flow.cast(input_def, dtype=dtype)
    input = np.random.rand(*input_shape).astype(oneflow2np_dtype_dict[dtype])
    of_out = cast_forward(input).get()
    tf_out = tf.cast(input, dtype=oneflow2np_dtype_dict[dtype])
    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=1e-5, atol=1e-5)


def compare_with_tensorflow(device_type, input_shape, input_dtype, dtype):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    @flow.function(func_config)
    def CastJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "in",
                shape=input_shape,
                dtype=input_dtype,
                initializer=flow.random_uniform_initializer(),
                trainable=True,
            )

            loss = flow.cast(x, dtype=flow.float)
            flow.losses.add_loss(loss)
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))
            return loss

     # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = CastJob().get()
    # TensorFlow
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(test_global_storage.Get("x"))
        tf_out = tf.cast(x, dtype=oneflow2np_dtype_dict[dtype])
    loss_diff = test_global_storage.Get("loss_diff")
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)
    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(test_global_storage.Get("x_diff"),
                       tf_x_diff.numpy(), rtol=1e-5, atol=1e-5)


def test_cast(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(5, 4, 3)]
    arg_dict["input_dtype"] = [flow.float, flow.double]
    arg_dict["dtype"] = [flow.float, flow.double]
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

def test_cast_forward(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu", "cpu"]
    arg_dict["input_shape"] = [(5, 4, 3)]
    arg_dict["dtype"] = [flow.float, flow.int8, flow.uint8, flow.double, flow.int32, flow.int64]
    arg_dict["switch_dtype"] = [flow.float, flow.int8, flow.uint8, flow.double, flow.int32, flow.int64]
    for arg in GenArgList(arg_dict):
        cast_forward_compare_with_tensorflow(test_case, *arg)


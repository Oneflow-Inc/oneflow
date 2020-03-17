import oneflow as flow
import numpy as np
import tensorflow as tf
import os
from collections import OrderedDict 

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save

def compare_with_tensorflow(device_type, input_shape, padding, output_shape, 
                            strides, kernel_size, dilations, out_channels, data_format):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))
    func_config.cudnn_conv_force_fwd_algo(0)
    func_config.cudnn_conv_force_bwd_data_algo(1)
    func_config.cudnn_conv_force_bwd_filter_algo(1)

    if data_format == 'NCHW':
        in_channels = input_shape[1]
    else:
        in_channels = input_shape[3]
    @flow.function(func_config)
    def DeconvJob():
        with flow.device_prior_placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=input_shape,
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            if data_format == 'NCHW':
                weight = flow.get_variable(
                    "weight",
                    shape=(in_channels, out_channels, kernel_size, kernel_size),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                    trainable=True,
                )
            else:
                weight = flow.get_variable(
                    "weight",
                    shape=(in_channels, kernel_size, kernel_size, out_channels),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                    trainable=True,
                )
            loss = flow.nn.conv2d_transpose_V2(x, weight, strides=strides, output_shape=output_shape, 
                                             dilations=dilations, padding=padding, data_format=data_format)
            flow.losses.add_loss(loss)

            flow.watch(x, Save("x"))
            flow.watch(weight, Save("weight"))
            flow.watch_diff(x, Save("x_diff"))
            flow.watch(loss, Save("loss"))
            flow.watch_diff(loss, Save("loss_diff"))

            return loss

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = DeconvJob().get()
    # Tensorflow
    tf.enable_eager_execution()
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(np.load(os.path.join(GetSavePath(), "x.npy")))
        if data_format == 'NCHW':
            w = tf.Variable(np.load(os.path.join(GetSavePath(), "weight.npy")).transpose((2,3,1,0)))
        else:
            w = tf.Variable(np.load(os.path.join(GetSavePath(), "weight.npy")).transpose((1,2,3,0)))
        tf_out = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1,1,strides,strides], padding="SAME",
                                        data_format=data_format)
    loss_diff = np.load(os.path.join(GetSavePath(), "loss_diff.npy"))
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "x_diff.npy")), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )

def test_deconv_with_tf_NHWC(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(2, 3, 3, 4)]
    arg_dict["padding"] = ["SAME"]   
    arg_dict["output_shape"] = [(2, 3, 3, 8)]   
    arg_dict["strides"] = [1]   
    arg_dict["kernel_size"] = [3]   
    arg_dict["dilations"] = [1]   
    arg_dict["out_channels"] = [8]
    arg_dict["data_format"] = ['NHWC']   
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

def test_deconv_with_tf_NCHW(test_case):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(2, 4, 3, 3)]
    arg_dict["padding"] = ["SAME"]   
    arg_dict["output_shape"] = [(2, 8, 6, 6)]   
    arg_dict["strides"] = [2]   
    arg_dict["kernel_size"] = [3, 4, 5]   
    arg_dict["dilations"] = [1]   
    arg_dict["out_channels"] = [8]
    arg_dict["data_format"] = ['NCHW']   
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

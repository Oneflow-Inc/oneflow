import oneflow as flow
import torch
import numpy as np
import tensorflow as tf
import os
from collections import OrderedDict 

from test_util import GenArgList
from test_util import GetSavePath
from test_util import Save



def compare_with_pytorch(device_type, input_shape, padding, output_padding, 
                         stride, kernel_size, dilations, out_channels):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    in_channels = input_shape[1]
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
            weight = flow.get_variable(
                "weight",
                shape=(in_channels, out_channels, kernel_size, kernel_size),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(),
                trainable=True,
            )
            loss = flow.nn.conv2d_transpose(x, weight, strides=stride, output_padding=output_padding, 
                                            dilations=dilations, padding=padding, data_format="NCHW")
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
    # Pytorch
    deconv = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, output_padding=output_padding, 
                                      groups=1, bias=False, dilation=dilations, padding_mode='zeros')
    deconv.weight = torch.nn.Parameter(torch.from_numpy(np.load(os.path.join(GetSavePath(), "weight.npy"))))
    torch_out = deconv(torch.from_numpy(np.load(os.path.join(GetSavePath(), "x.npy"))))
    assert np.allclose(of_out.ndarray(), torch_out.detach().numpy(), rtol=1e-5, atol=1e-5)
    # TODO: compare backward

def compare_with_tensorflow(device_type, input_shape, padding, output_shape, 
                            strides, kernel_size, dilations, out_channels):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.train.primary_lr(1e-4)
    func_config.train.model_update_conf(dict(naive_conf={}))

    in_channels = input_shape[1]
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
            weight = flow.get_variable(
                "weight",
                shape=(in_channels, out_channels, kernel_size, kernel_size),
                dtype=flow.float,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            loss = flow.nn.conv2d_transpose_V2(x, weight, strides=strides, output_shape=output_shape, 
                                             dilations=dilations, padding=padding, data_format="NCHW")
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
    with tf.GradientTape(persistent=True) as tape:
        x = tf.Variable(np.load(os.path.join(GetSavePath(), "x.npy")))
        w = tf.Variable(np.load(os.path.join(GetSavePath(), "weight.npy")).transpose((2,3,1,0)))
        tf_out = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1,1,strides,strides], padding="SAME",
                                        data_format="NCHW")
    loss_diff = np.load(os.path.join(GetSavePath(), "loss_diff.npy"))
    tf_x_diff = tape.gradient(tf_out, x, loss_diff)

    assert np.allclose(of_out.ndarray(), tf_out.numpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(
        np.load(os.path.join(GetSavePath(), "x_diff.npy")), tf_x_diff.numpy(), rtol=1e-5, atol=1e-5
    )

def test_deconv_with_torch():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(2, 1, 3, 3)]
    arg_dict["padding"] = [1]   
    arg_dict["output_padding"] = [1]   
    arg_dict["strides"] = [2]   
    arg_dict["kernel_size"] = [5]   
    arg_dict["dilations"] = [2]   
    arg_dict["out_channels"] = [4]   
    for arg in GenArgList(arg_dict):
        compare_with_pytorch(*arg)

def test_deconv_with_tf():
    arg_dict = OrderedDict()
    arg_dict["device_type"] = ["gpu"]
    arg_dict["input_shape"] = [(2, 1, 3, 3)]
    arg_dict["padding"] = ["SAME"]   
    arg_dict["output_shape"] = [(2, 1, 3, 3)]   
    arg_dict["strides"] = [1]   
    arg_dict["kernel_size"] = [5]   
    arg_dict["dilations"] = [1]   
    arg_dict["out_channels"] = [1]   
    for arg in GenArgList(arg_dict):
        compare_with_tensorflow(*arg)

if __name__ == "__main__":
    tf.enable_eager_execution()
    # test_deconv_with_tf()
    test_deconv_with_torch()

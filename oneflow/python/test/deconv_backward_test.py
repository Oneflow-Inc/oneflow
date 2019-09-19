import oneflow as flow
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.common.data_type_pb2 as data_type_util
from datetime import datetime
import argparse
import numpy as np
import os

def _get_initializer():
    kernel_initializer = op_conf_util.InitializerConf()
    kernel_initializer.truncated_normal_conf.std = 0.816496580927726
    return kernel_initializer

def _conv2d_layer(
    name,
    input,
    filters,
    kernel_size=3,
    strides=1,
    padding="SAME",
    data_format="NCHW",
    dilation_rate=1,
    activation=op_conf_util.kRelu,
    use_bias=False,
    weight_initializer=flow.random_uniform_initializer(),
    bias_initializer=None,
):
    weight_shape = (filters, input.static_shape[1], kernel_size, kernel_size)
    weight = flow.get_variable(
        name + "-weight",
        shape=weight_shape,
        dtype=input.dtype,
        initializer=weight_initializer,
    )
    output = flow.nn.conv2d(
        input, weight, strides, padding, data_format, dilation_rate, name=name
    )
    if use_bias:
        bias = flow.get_variable(
            name + "-bias",
            shape=(filters,),
            dtype=input.dtype,
            initializer=bias_initializer,
        )
        output = flow.nn.bias_add(output, bias, data_format)

    if activation is not None:
        if activation == op_conf_util.kRelu:
            output = flow.keras.activations.relu(output)
        else:
            raise NotImplementedError

    return output

def Print(x):
    print("x: ")
    print(x)

@flow.function
def DeconvBackwardTestJob(x = flow.input_blob_def((64, 256, 14, 14)), 
                          labels = flow.input_blob_def((64, 1), dtype=data_type_util.kInt32)):

    flow.config.train.primary_lr(0.1)
    flow.config.train.model_update_conf(dict(naive_conf={}))

    flow.config.cudnn_conv_force_fwd_algo(0)
    flow.config.cudnn_conv_force_bwd_data_algo(1)
    flow.config.cudnn_conv_force_bwd_filter_algo(1)

    # test net work
    conv1 = _conv2d_layer(
        "conv1", x, filters=256, kernel_size=3, strides=1, padding="SAME"
    )
    filter = flow.get_variable(name="filter", shape=(
        256, 3, 2, 2), dtype=flow.float32, initializer=_get_initializer())
    deconv = flow.nn.conv2d_transpose(conv1, filter, strides=2, data_format="NCHW")

    flat = flow.reshape(deconv, shape=(deconv.static_shape[0], -1))
    fc = flow.layers.dense(
        inputs=flat,
        units=4,
        activation=flow.keras.activations.relu,
        use_bias=False,
        kernel_initializer=_get_initializer(),
        bias_initializer=False,
        trainable=True,
        name="fc",
    )

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, fc, name="softmax_loss")
    flow.losses.add_loss(loss)
    flow.watch(filter, Print)
    return filter


if __name__ == "__main__":

    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float32)
    check_point = flow.train.CheckPoint()
    check_point.init()
    for i in range(2):
        x = np.random.randn(64, 256, 14, 14).astype(np.float32)
        label = np.random.randint(4, size=64).astype(np.int32)
        filter = DeconvBackwardTestJob(x, label)
        print(filter)

"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import logging

import numpy as np
from onnx import onnx_pb
from onnx.onnx_pb import TensorProto
from oneflow.python.framework import id_util
from oneflow.python.onnx import constants, util
from oneflow.python.onnx.handler import flow_op

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable


def _SpatialMap(shape, perm):
    new_shape = shape[:]
    for i in perm:
        new_shape[i] = shape[perm[i]]
    return new_shape


def _ConvConvertInputs(
    ctx,
    node,
    with_kernel=False,
    new_kernel_shape=None,
    input_indices=None,
    output_indices=None,
):
    """Convert input and kernel from oneflow to onnx. This maybe require to
        to insert transpose ops for input, kernel and output unless they are constants
        and we can transpose the constant.
        We transpose inputs and the kernel if the input is in NHWC
        Outputs are transposed if the format is NHWC.
        Some convolutions like depthwise_conv2d require a reshape of the kernel.
        Args:
            ctx: the parent graph
            node: node of the convolution op
            with_kernel: transpose the kernel
            new_kernel_shape: reshape the kernel
    """

    if input_indices is None:
        input_indices = [0]
    if output_indices is None:
        output_indices = [0]

    if node.is_nhwc():
        # transpose input if needed, no need to record shapes on input
        for idx in input_indices:
            parent = node.inputs[idx]
            if (
                node.inputs[idx].is_const()
                and len(ctx.FindOutputConsumers(node.input[1])) == 1
            ):
                # if input is a constant, transpose that one if we are the only consumer
                val = parent.get_tensor_value(as_list=False)
                parent.set_tensor_value(val.transpose(constants.NHWC_TO_NCHW))
            else:
                # if input comes from a op, insert transpose op
                input_name = node.input[idx]
                transpose = ctx.InsertNewNodeOnInput(node, "Transpose", input_name)
                transpose.set_attr("perm", constants.NHWC_TO_NCHW)
                transpose.skip_conversion = True
                shape = ctx.get_shape(input_name)
                if shape is not None:
                    new_shape = _SpatialMap(shape, constants.NHWC_TO_NCHW)
                    ctx.set_shape(transpose.output[0], new_shape)

    # kernel need to be transposed if the data format is nhwc
    if with_kernel:
        # some onnx conv ops require the reshape the kernel (ie. depthwise_conv2d)
        if new_kernel_shape:
            if ctx.opset < 5:
                # old reshape takes new shape as attribute
                input_name = node.input[1]
                reshape = ctx.InsertNewNodeOnInput(node, "Reshape", input_name)
                reshape.set_attr("shape", new_kernel_shape)
                reshape.skip_conversion = True
            else:
                # new reshape takes new shape as input[1]
                shape_name = id_util.UniqueStr(node.name)
                ctx.MakeConst(shape_name, np.array(new_kernel_shape, dtype=np.int64))
                input_name = node.input[1]
                reshape = ctx.MakeNode("Reshape", [input_name, shape_name])
                ctx.ReplaceAllInputs(node, input_name, reshape.output[0])
                reshape.skip_conversion = True
            ctx.set_shape(reshape.output[0], new_kernel_shape)

        if node.is_nhwc():
            parent = node.inputs[1]
            need_transpose = True
            if node.inputs[1].is_const():
                # kernel is const - transpose the const if we are the only consumer of const
                consumers = ctx.FindOutputConsumers(node.input[1])
                if len(consumers) == 1:
                    val = parent.get_tensor_value(as_list=False)
                    val = val.transpose(constants.NHWC_TO_NCHW)
                    parent.set_tensor_value(val)
                    need_transpose = False

            if need_transpose:
                input_name = node.input[1]
                transpose = ctx.InsertNewNodeOnInput(node, "Transpose", input_name)
                transpose.set_attr("perm", constants.NHWC_TO_NCHW)
                transpose.skip_conversion = True
                new_shape = _SpatialMap(
                    ctx.get_shape(input_name), constants.NHWC_TO_NCHW
                )
                ctx.set_shape(transpose.output[0], new_shape)

    # transpose outputs if needed
    if node.is_nhwc():
        for idx in output_indices:
            output_name = node.output[idx]
            output_shape = ctx.get_shape(node.output[idx])
            op_name = id_util.UniqueStr(node.name)
            transpose = ctx.InsertNewNodeOnOutput(
                "Transpose", output_name, name=op_name
            )
            transpose.set_attr("perm", constants.NCHW_TO_NHWC)
            transpose.skip_conversion = True
            # set NHWC shape to transpose node output
            ctx.set_shape(transpose.output[0], output_shape)
            # Transpose NHWC shape back to NCHW shape for current ONNX conv node output
            ctx.set_shape(
                output_name, _SpatialMap(output_shape, constants.NHWC_TO_NCHW)
            )
        node.data_format = "NCHW"


def conv_dims_attr(node, name, new_name=None):
    if new_name is None:
        new_name = name
    dims = node.get_attr(name)
    if not dims:
        return None
    dims = dims.ints
    if len(dims) == 2:
        h, w = dims
        c = n = 1
    else:
        if node.is_nhwc():
            n, h, w, c = dims
        else:
            n, c, h, w = dims
    dims = [h, w]
    node.set_attr(new_name, dims)
    return dims


def conv_kernel_shape(ctx, node, input_idx, spatial=2):
    kernel_shape = node.get_attr("kernel_size").ints
    node.set_attr("kernel_shape", kernel_shape)
    return kernel_shape


@flow_op(["conv2d"], flow_ibns=["in", "weight"])
class ConvOp:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # T output = Conv2D(T input, T filter, @list(int) strides, @bool use_cudnn_on_gpu,
        #                       @string padding, @string data_format)
        # T Y = Conv(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
        #                       @AttrType.INTS kernel_shape, @AttrType.INTS pads, @AttrType.INTS strides)
        node.type = "Conv"
        kernel_shape = conv_kernel_shape(ctx, node, 1, spatial=2)
        node.set_attr("group", node.get_attr_value("groups", 1))
        node.set_attr("dilations", node.get_attr_value("dilation_rate", [1, 1]))
        strides = conv_dims_attr(node, "strides")
        dilations = conv_dims_attr(node, "dilations")
        node.set_attr("pads", node.get_attr_value("padding_before", [0, 0]) * 2)
        _ConvConvertInputs(ctx, node, with_kernel=True)

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # no change
        cls.Version_1(ctx, node, **kwargs)


@flow_op(["avg_pool_2d"], onnx_op="AveragePool")
@flow_op(["max_pool_2d"], onnx_op="MaxPool")
class PoolOp:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        cls._Convert(ctx, node, **kwargs)

    @classmethod
    def Version_10(cls, ctx, node, **kwargs):
        cls._Convert(ctx, node, **kwargs)

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # no change
        cls._Convert(ctx, node, **kwargs)

    @classmethod
    def _Convert(cls, ctx, node, **kwargs):
        # T output = MaxPool(T input, @list(int) ksize, @list(int) strides, @string padding, @string data_format)
        # T Y = MaxPool(T X, @AttrType.STRING auto_pad, @AttrType.INTS kernel_shape, @AttrType.INTS pads,
        #               @AttrType.INTS strides)
        if len(node.input) < 3:
            kernel_shape_flow = node.get_attr("pool_size").ints
            strides_flow = node.get_attr("strides").ints
        else:
            kernel_shape_flow = node.inputs[1].get_tensor_value()
            strides_flow = node.inputs[2].get_tensor_value()
            ctx.RemoveInput(node, node.input[2])
            ctx.RemoveInput(node, node.input[1])

        node.set_attr("kernel_shape", kernel_shape_flow)
        node.set_attr("strides", strides_flow)
        conv_dims_attr(node, "dilations")
        node.set_attr("pads", node.get_attr_value("padding_before", [0, 0]) * 2)
        _ConvConvertInputs(ctx, node, with_kernel=False)


@flow_op(["pad"], onnx_op="Pad")
class Pad:
    @classmethod
    def Version_2(cls, ctx, node, **kwargs):
        padding_before = node.get_attr_value("padding_before")
        padding_after = node.get_attr_value("padding_after")
        paddings = padding_before + padding_after
        node.set_attr("pads", paddings)
        node.set_attr("mode", "constant")
        const_val = (
            node.get_attr_value("integral_constant_value")
            if util.is_integral_onnx_dtype(ctx.get_dtype(node.input[0]))
            else node.get_attr_value("floating_constant_value")
        )
        node.set_attr("value", const_val)

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        node.set_attr("mode", "constant")
        padding_before = node.get_attr_value("padding_before")
        padding_after = node.get_attr_value("padding_after")
        paddings = np.array(padding_before + padding_after).astype(np.int64)
        padding_node = ctx.MakeConst(id_util.UniqueStr("const"), paddings)
        node.input.append(padding_node.output[0])
        dtype = ctx.get_dtype(node.input[0])
        const_val = (
            node.get_attr_value("integral_constant_value")
            if util.is_integral_onnx_dtype(dtype)
            else node.get_attr_value("floating_constant_value")
        )
        const_val = np.array(const_val).astype(util.Onnx2NumpyDtype(dtype))
        const_val_node = ctx.MakeConst(id_util.UniqueStr("const"), const_val)
        node.input.append(const_val_node.output[0])


@flow_op(
    ["normalization"],
    flow_ibns=["x", "gamma", "beta", "moving_mean", "moving_variance"],
)
class BatchNorm:
    @classmethod
    def Version_6(cls, ctx, node, **kwargs):
        node.type = "BatchNormalization"
        # flow inputs: x, gamma, beta, moving_mean, moving_variance
        # flow outputs: y, mean, inv_variance
        # a: data_format, epsilon, is_training
        # onnx inputs: X, scale, B, mean, variance, attributes: epsilon, momentum=0.9, spatial : 1
        # output: y, mean, var, savedmean, savedvar,
        # detach unused outputs. While we could let the unused outputs dangle,
        # some runtimes like pytorch/caffe2 do complain about it.
        if node.get_attr_value("training") or node.get_attr_value("trainable"):
            raise NotImplementedError(
                "We only support inference mode ONNX BatchNormalization now"
            )
        consumers = [
            ctx.FindOutputConsumers(output_name) for output_name in node.output[1:]
        ]
        if not any(consumers):
            new_output = [node.output[0]]
            node.output = new_output

        _ConvConvertInputs(ctx, node, with_kernel=False)

        scale_shape = ctx.get_shape(node.input[1])
        mean_shape = ctx.get_shape(node.input[3])
        var_shape = ctx.get_shape(node.input[4])
        val_type = util.Onnx2NumpyDtype(ctx.get_dtype(node.input[1]))

        if mean_shape != scale_shape:
            new_mean_value = np.array(
                np.resize(node.inputs[3].get_tensor_value(as_list=False), scale_shape),
                dtype=val_type,
            )
            new_mean_node_name = id_util.UniqueStr(node.name)
            ctx.MakeConst(new_mean_node_name, new_mean_value)
            node.input[3] = new_mean_node_name

        if var_shape != scale_shape:
            new_var_value = np.array(
                np.resize(node.inputs[4].get_tensor_value(as_list=False), scale_shape),
                dtype=val_type,
            )
            new_val_node_name = id_util.UniqueStr(node.name)
            ctx.MakeConst(new_val_node_name, new_var_value)
            node.input[4] = new_val_node_name

    @classmethod
    def Version_9(cls, ctx, node, **kwargs):
        # is_test was removed - no change for us
        cls.Version_6(ctx, node, **kwargs)

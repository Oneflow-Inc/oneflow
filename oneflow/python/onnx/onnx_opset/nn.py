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
            parent = node.input_nodes[idx]
            if (
                node.input_nodes[idx].is_const()
                and len(ctx.FindOutputConsumers(node.input_tensor_names[1])) == 1
            ):
                # if input is a constant, transpose that one if we are the only consumer
                val = parent.get_tensor_value(as_list=False)
                parent.set_tensor_value(val.transpose(constants.NHWC_TO_NCHW))
            else:
                # if input comes from a op, insert transpose op
                input_name = node.input_tensor_names[idx]
                transpose = ctx.InsertNewNodeOnInput(node, "Transpose", input_name)
                transpose.attrs["perm"] = constants.NHWC_TO_NCHW
                transpose.skip_conversion = True
                shape = ctx.get_shape(input_name)
                if shape is not None:
                    new_shape = _SpatialMap(shape, constants.NHWC_TO_NCHW)
                    ctx.set_shape(transpose.output_tensor_names[0], new_shape)

    # kernel need to be transposed if the data format is nhwc
    if with_kernel:
        # some onnx conv ops require the reshape the kernel (ie. depthwise_conv2d)
        if new_kernel_shape:
            if ctx.opset < 5:
                # old reshape takes new shape as attribute
                input_name = node.input_tensor_names[1]
                reshape = ctx.InsertNewNodeOnInput(node, "Reshape", input_name)
                reshape.attrs["shape"] = new_kernel_shape
                reshape.skip_conversion = True
            else:
                # new reshape takes new shape as input_tensor_names[1]
                shape_name = id_util.UniqueStr(node.name)
                ctx.MakeConst(shape_name, np.array(new_kernel_shape, dtype=np.int64))
                input_name = node.input_tensor_names[1]
                reshape = ctx.MakeNode("Reshape", [input_name, shape_name])
                ctx.ReplaceAllInputs(node, input_name, reshape.output_tensor_names[0])
                reshape.skip_conversion = True
            ctx.set_shape(reshape.output_tensor_names[0], new_kernel_shape)

        if node.is_nhwc():
            parent = node.input_nodes[1]
            need_transpose = True
            if node.input_nodes[1].is_const():
                # kernel is const - transpose the const if we are the only consumer of const
                consumers = ctx.FindOutputConsumers(node.input_tensor_names[1])
                if len(consumers) == 1:
                    val = parent.get_tensor_value(as_list=False)
                    val = val.transpose(constants.NHWC_TO_NCHW)
                    parent.set_tensor_value(val)
                    need_transpose = False

            if need_transpose:
                input_name = node.input_tensor_names[1]
                transpose = ctx.InsertNewNodeOnInput(node, "Transpose", input_name)
                transpose.attrs["perm"] = constants.NHWC_TO_NCHW
                transpose.skip_conversion = True
                new_shape = _SpatialMap(
                    ctx.get_shape(input_name), constants.NHWC_TO_NCHW
                )
                ctx.set_shape(transpose.output_tensor_names[0], new_shape)

    # transpose outputs if needed
    if node.is_nhwc():
        for idx in output_indices:
            output_name = node.output_tensor_names[idx]
            output_shape = ctx.get_shape(node.output_tensor_names[idx])
            op_name = id_util.UniqueStr(node.name)
            transpose = ctx.InsertNewNodeOnOutput(
                "Transpose", output_name, name=op_name
            )
            transpose.attrs["perm"] = constants.NCHW_TO_NHWC
            transpose.skip_conversion = True
            # set NHWC shape to transpose node output_tensor_names
            ctx.set_shape(transpose.output_tensor_names[0], output_shape)
            # Transpose NHWC shape back to NCHW shape for current ONNX conv node output
            ctx.set_shape(
                output_name, _SpatialMap(output_shape, constants.NHWC_TO_NCHW)
            )
        node.data_format = "NCHW"


def _AddPadding(ctx, node, kernel_shape, strides, dilations=None, spatial=2):
    padding = node.attrs.get("padding")
    if padding:
        if dilations is None:
            dilations = [1] * spatial * 2
        if padding == "same":
            padding = "same_lower"
        if padding in ["same_lower", "same_upper"]:
            pads = [0] * spatial * 2
            input_shape = ctx.get_shape(node.input_tensor_names[0])
            output_shape = ctx.get_shape(node.output_tensor_names[0])
            # check if the input shape is valid
            if len(input_shape) != len(pads):
                logger.error(
                    "node %s input needs to be rank %d, is %d",
                    node.name,
                    len(pads),
                    len(input_shape),
                )
            # transpose shape to nchw
            if node.is_nhwc():
                input_shape = _SpatialMap(input_shape, constants.NHWC_TO_NCHW)
                output_shape = _SpatialMap(output_shape, constants.NHWC_TO_NCHW)
            # calculate pads
            if any(
                input_shape[i + 2] == -1 or output_shape[i + 2] == -1
                for i in range(spatial)
            ):
                logger.debug(
                    "node %s has unknown dim for pads calculation, fallback to auto_pad: "
                    "input_shape=%s, output_shape=%s",
                    node.name,
                    input_shape,
                    output_shape,
                )
                if padding == "same_lower":
                    node.attrs["auto_pad"] = "SAME_LOWER"
                else:
                    node.attrs["auto_pad"] = "SAME_UPPER"
            else:
                for i in range(spatial):
                    pad = (
                        (output_shape[i + 2] - 1) * strides[i]
                        + dilations[i] * (kernel_shape[i] - 1)
                        + 1
                        - input_shape[i + 2]
                    )
                    pad = max(pad, 0)
                    if padding == "same_lower":
                        pads[i + spatial] = pad // 2
                        pads[i] = pad - pad // 2
                    else:
                        pads[i] = pad // 2
                        pads[i + spatial] = pad - pad // 2
                node.attrs["pads"] = pads

        elif padding == "valid":
            pass
        else:
            raise ValueError("invalid padding value: " + padding)


def conv_dims_attr(node, name, new_name=None):
    if new_name is None:
        new_name = name
    dims = node.attrs.get(name, None)
    if not dims:
        return None
    if len(dims) == 2:
        h, w = dims
        c = n = 1
    else:
        if node.is_nhwc():
            n, h, w, c = dims
        else:
            n, c, h, w = dims
    dims = [h, w]
    node.attrs[new_name] = dims
    return dims


def conv_kernel_shape(ctx, node, input_idx, spatial=2):
    node.attrs["kernel_shape"] = node.attrs["kernel_size"]
    return node.attrs["kernel_shape"]


@flow_op(["conv2d"], flow_ibns=["in", "weight"])
class ConvOp:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # T output = Conv2D(T input, T filter, @list(int) strides, @bool use_cudnn_on_gpu,
        #                       @string padding, @string data_format)
        # T Y = Conv(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
        #                       @AttrType.INTS kernel_shape, @AttrType.INTS pads, @AttrType.INTS strides)
        node.op_type = "Conv"
        kernel_shape = conv_kernel_shape(ctx, node, 1, spatial=2)
        node.attrs["group"] = node.attrs.get("groups", 1)
        node.attrs["dilations"] = node.attrs.get("dilation_rate", [1, 1])
        strides = conv_dims_attr(node, "strides")
        dilations = conv_dims_attr(node, "dilations")
        node.attrs["pads"] = node.attrs.get("padding_before", [0, 0]) * 2
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
        if len(node.input_tensor_names) < 3:
            kernel_shape_flow = node.attrs["pool_size"]
            strides_flow = node.attrs["strides"]
        else:
            kernel_shape_flow = node.input_nodes[1].get_tensor_value()
            strides_flow = node.input_nodes[2].get_tensor_value()
            ctx.RemoveInput(node, node.input_tensor_names[2])
            ctx.RemoveInput(node, node.input_tensor_names[1])

        node.attrs["kernel_shape"] = kernel_shape_flow
        node.attrs["strides"] = strides_flow
        conv_dims_attr(node, "dilations")
        if "padding" in node.attrs:
            _AddPadding(ctx, node, kernel_shape_flow, strides_flow)
        else:
            pads = node.attrs.get("padding_before", [0, 0]) + node.attrs.get(
                "padding_after", [0, 0]
            )
            node.attrs["pads"] = pads
        _ConvConvertInputs(ctx, node, with_kernel=False)


@flow_op(["pad"], onnx_op="Pad")
class Pad:
    @classmethod
    def Version_2(cls, ctx, node, **kwargs):
        padding_before = node.attrs["padding_before"]
        padding_after = node.attrs["padding_after"]
        paddings = padding_before + padding_after
        node.attrs["pads"] = paddings
        node.attrs["mode"] = "constant"
        const_val = (
            node.attrs["integral_constant_value"]
            if util.is_integral_onnx_dtype(ctx.get_dtype(node.input_tensor_names[0]))
            else node.attrs["floating_constant_value"]
        )
        node.attrs["value"] = const_val

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        node.attrs["mode"] = "constant"
        padding_before = node.attrs["padding_before"]
        padding_after = node.attrs["padding_after"]
        paddings = np.array(padding_before + padding_after).astype(np.int64)
        padding_node = ctx.MakeConst(id_util.UniqueStr("const"), paddings)
        node.input_tensor_names.append(padding_node.output_tensor_names[0])
        dtype = ctx.get_dtype(node.input_tensor_names[0])
        const_val = (
            node.attrs["integral_constant_value"]
            if util.is_integral_onnx_dtype(dtype)
            else node.attrs["floating_constant_value"]
        )
        const_val = np.array(const_val).astype(util.Onnx2NumpyDtype(dtype))
        const_val_node = ctx.MakeConst(id_util.UniqueStr("const"), const_val)
        node.input_tensor_names.append(const_val_node.output_tensor_names[0])


@flow_op(
    ["normalization"],
    flow_ibns=["x", "gamma", "beta", "moving_mean", "moving_variance"],
)
class BatchNorm:
    @classmethod
    def Version_6(cls, ctx, node, **kwargs):
        node.op_type = "BatchNormalization"
        # flow inputs: x, gamma, beta, moving_mean, moving_variance
        # flow outputs: y, mean, inv_variance
        # a: data_format, epsilon, is_training
        # onnx inputs: X, scale, B, mean, variance, attributes: epsilon, momentum=0.9, spatial : 1
        # output: y, mean, var, savedmean, savedvar,
        # detach unused outputs. While we could let the unused outputs dangle,
        # some runtimes like pytorch/caffe2 do complain about it.
        if node.attrs["training"]:
            raise NotImplementedError(
                "We only support inference mode ONNX BatchNormalization now"
            )
        consumers = [
            ctx.FindOutputConsumers(output_name)
            for output_name in node.output_tensor_names[1:]
        ]
        if not any(consumers):
            new_output = [node.output_tensor_names[0]]
            node.output_tensor_names = new_output

        _ConvConvertInputs(ctx, node, with_kernel=False)

        scale_shape = ctx.get_shape(node.input_tensor_names[1])
        mean_shape = ctx.get_shape(node.input_tensor_names[3])
        var_shape = ctx.get_shape(node.input_tensor_names[4])
        val_type = util.Onnx2NumpyDtype(ctx.get_dtype(node.input_tensor_names[1]))

        if mean_shape != scale_shape:
            new_mean_value = np.array(
                np.resize(
                    node.input_nodes[3].get_tensor_value(as_list=False), scale_shape
                ),
                dtype=val_type,
            )
            new_mean_node_name = id_util.UniqueStr(node.name)
            ctx.MakeConst(new_mean_node_name, new_mean_value)
            node.input_tensor_names[3] = new_mean_node_name

        if var_shape != scale_shape:
            new_var_value = np.array(
                np.resize(
                    node.input_nodes[4].get_tensor_value(as_list=False), scale_shape
                ),
                dtype=val_type,
            )
            new_val_node_name = id_util.UniqueStr(node.name)
            ctx.MakeConst(new_val_node_name, new_var_value)
            node.input_tensor_names[4] = new_val_node_name

    @classmethod
    def Version_9(cls, ctx, node, **kwargs):
        # is_test was removed - no change for us
        cls.Version_6(ctx, node, **kwargs)

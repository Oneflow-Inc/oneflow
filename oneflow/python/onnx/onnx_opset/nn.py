# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
nn
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from onnx import onnx_pb
from onnx.onnx_pb import TensorProto
from oneflow.python.onnx import constants, utils
from oneflow.python.onnx.graph_builder import GraphBuilder
from oneflow.python.onnx.handler import flow_op
from oneflow.python.onnx.onnx_opset import common, controlflow, tensor

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable

def spatial_map(shape, perm):
    new_shape = shape[:]
    for i in perm:
        new_shape[i] = shape[perm[i]]
    return new_shape


def conv_convert_inputs(ctx, node, with_kernel=False, new_kernel_shape=None,
                        input_indices=None, output_indices=None):
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
            if node.inputs[idx].is_const() and len(ctx.find_output_consumers(node.input[1])) == 1:
                # if input is a constant, transpose that one if we are the only consumer
                val = parent.get_tensor_value(as_list=False)
                parent.set_tensor_value(val.transpose(constants.NHWC_TO_NCHW))
            else:
                # if input comes from a op, insert transpose op
                input_name = node.input[idx]
                transpose = ctx.insert_new_node_on_input(
                    node, "Transpose", input_name)
                transpose.set_attr("perm", constants.NHWC_TO_NCHW)
                transpose.skip_conversion = True
                shape = ctx.get_shape(input_name)
                if shape is not None:
                    new_shape = spatial_map(shape, constants.NHWC_TO_NCHW)
                    ctx.set_shape(transpose.output[0], new_shape)

    # kernel need to be transposed if the data format is nhwc
    if with_kernel:
        # some onnx conv ops require the reshape the kernel (ie. depthwise_conv2d)
        if new_kernel_shape:
            if ctx.opset < 5:
                # old reshape takes new shape as attribute
                input_name = node.input[1]
                reshape = ctx.insert_new_node_on_input(
                    node, "Reshape", input_name)
                reshape.set_attr("shape", new_kernel_shape)
                reshape.skip_conversion = True
            else:
                # new reshape takes new shape as input[1]
                shape_name = utils.make_name(node.name)
                ctx.make_const(shape_name, np.array(
                    new_kernel_shape, dtype=np.int64))
                input_name = node.input[1]
                reshape = ctx.make_node("Reshape", [input_name, shape_name])
                ctx.replace_input(node, input_name, reshape.output[0])
                reshape.skip_conversion = True
            ctx.set_shape(reshape.output[0], new_kernel_shape)

        if node.is_nhwc():
            parent = node.inputs[1]
            need_transpose = True
            if node.inputs[1].is_const():
                # kernel is const - transpose the const if we are the only consumer of const
                consumers = ctx.find_output_consumers(node.input[1])
                if len(consumers) == 1:
                    val = parent.get_tensor_value(as_list=False)
                    import pdb
                    pdb.set_trace()
                    val = val.transpose(constants.NHWC_TO_NCHW)
                    parent.set_tensor_value(val)
                    need_transpose = False

            if need_transpose:
                input_name = node.input[1]
                transpose = ctx.insert_new_node_on_input(
                    node, "Transpose", input_name)
                transpose.set_attr("perm", constants.NHWC_TO_NCHW)
                transpose.skip_conversion = True
                new_shape = spatial_map(ctx.get_shape(
                    input_name), constants.NHWC_TO_NCHW)
                ctx.set_shape(transpose.output[0], new_shape)

    # transpose outputs if needed
    if node.is_nhwc():
        for idx in output_indices:
            output_name = node.output[idx]
            output_shape = ctx.get_shape(node.output[idx])
            op_name = utils.make_name(node.name)
            transpose = ctx.insert_new_node_on_output(
                "Transpose", output_name, name=op_name)
            transpose.set_attr("perm", constants.NCHW_TO_NHWC)
            transpose.skip_conversion = True
            # set TF NHWC shape to transpose node output
            ctx.set_shape(transpose.output[0], output_shape)
            # Transpose TF NHWC shape back to NCHW shape for current ONNX conv node output
            ctx.set_shape(output_name, spatial_map(
                output_shape, constants.NHWC_TO_NCHW))
        node.data_format = "NCHW"


def add_padding(ctx, node, kernel_shape, strides, dilations=None, spatial=2):
    padding = node.get_attr("padding")
    if padding:
        if dilations is None:
            dilations = [1] * spatial * 2
        padding = padding.s.decode("utf-8")
        if padding == 'same':
            pads = [0] * spatial * 2
            input_shape = ctx.get_shape(node.input[0])
            output_shape = ctx.get_shape(node.output[0])
            # check if the input shape is valid
            if len(input_shape) != len(pads):
                logger.error("node %s input needs to be rank %d, is %d",
                             node.name, len(pads), len(input_shape))
            # transpose shape to nchw
            if node.is_nhwc():
                input_shape = spatial_map(input_shape, constants.NHWC_TO_NCHW)
                output_shape = spatial_map(
                    output_shape, constants.NHWC_TO_NCHW)
            # calculate pads
            if any(input_shape[i + 2] == -1 or output_shape[i + 2] == -1 for i in range(spatial)):
                logger.debug(
                    "node %s has unknown dim for pads calculation, fallback to auto_pad: "
                    "input_shape=%s, output_shape=%s",
                    node.name, input_shape, output_shape)
                node.set_attr("auto_pad", "SAME_LOWER")
            else:
                for i in range(spatial):
                    pad = (output_shape[i + 2] - 1) * strides[i] + dilations[i] * \
                        (kernel_shape[i] - 1) + 1 - input_shape[i + 2]
                    pad = max(pad, 0)
                    # pads[i] = pad // 2
                    # pads[i + spatial] = pad - pad // 2
                    pads[i + spatial] = pad // 2
                    pads[i] = pad - pad // 2
                node.set_attr("pads", pads)

        elif padding == 'valid':
            pass
        else:
            raise ValueError("invalid padding value: " + padding)


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
    # kernel_shape = ctx.get_shape(node.input[input_idx])
    # if len(kernel_shape) != 2 * spatial:
    #     raise ValueError("kernel rank must be 2* spatial")
    # kernel_shape = kernel_shape[0:spatial]
    kernel_shape = node.get_attr('kernel_size').ints
    # import pdb; pdb.set_trace()
    node.set_attr("kernel_shape", kernel_shape)
    return kernel_shape


@flow_op(["conv2d"], flow_inputs=['in', 'weight'])
class ConvOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # T output = Conv2D(T input, T filter, @list(int) strides, @bool use_cudnn_on_gpu,
        #                       @string padding, @string data_format)
        # T Y = Conv(T X, T W, T B, @AttrType.STRING auto_pad, @AttrType.INTS dilations, @AttrType.INT group,
        #                       @AttrType.INTS kernel_shape, @AttrType.INTS pads, @AttrType.INTS strides)
        node.type = "Conv"
        kernel_shape = conv_kernel_shape(ctx, node, 1, spatial=2)
        node.set_attr('group', node.get_attr_value('groups', 1))
        node.set_attr('dilations', node.get_attr_value(
            'dilation_rate', [1, 1]))
        strides = conv_dims_attr(node, "strides")
        dilations = conv_dims_attr(node, "dilations")
        add_padding(ctx, node, kernel_shape, strides,
                    dilations=dilations, spatial=2)
        conv_convert_inputs(ctx, node, with_kernel=True)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls.version_1(ctx, node, **kwargs)


@flow_op(["avg_pool_2d"], onnx_op="AveragePool")
@flow_op(["max_pool_2d"], onnx_op="MaxPool")
class PoolOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        cls._convert(ctx, node, **kwargs)

    @classmethod
    def version_10(cls, ctx, node, **kwargs):
        cls._convert(ctx, node, **kwargs)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # no change
        cls._convert(ctx, node, **kwargs)

    @classmethod
    def _convert(cls, ctx, node, **kwargs):
        # T output = MaxPool(T input, @list(int) ksize, @list(int) strides, @string padding, @string data_format)
        # T Y = MaxPool(T X, @AttrType.STRING auto_pad, @AttrType.INTS kernel_shape, @AttrType.INTS pads,
        #               @AttrType.INTS strides)
        # above seems wrong - input[1] is ksize, input[2] is strides
        # stride and ksize in tf is not always NHWC, so watch out when converting into onnx's NCHW
        if len(node.input) < 3:
            kernel_shape_tf = node.get_attr("pool_size").ints
            strides_tf = node.get_attr("strides").ints
        else:
            kernel_shape_tf = node.inputs[1].get_tensor_value()
            strides_tf = node.inputs[2].get_tensor_value()
            ctx.remove_input(node, node.input[2])
            ctx.remove_input(node, node.input[1])

        node.set_attr("kernel_shape", kernel_shape_tf)
        node.set_attr("strides", strides_tf)
        conv_dims_attr(node, "dilations")
        add_padding(ctx, node, kernel_shape_tf, strides_tf)
        conv_convert_inputs(ctx, node, with_kernel=False)


@flow_op(["pad"], onnx_op="Pad")
class Pad:
    @classmethod
    def version_2(cls, ctx, node, **kwargs):
        padding_before = node.get_attr_value('padding_before')
        padding_after = node.get_attr_value('padding_after')
        paddings = padding_before + padding_after
        node.set_attr('pads', paddings)
        node.set_attr('mode', 'constant')
        const_val = node.get_attr_value('integral_constant_value') if utils.is_integral_onnx_dtype(
            ctx.get_dtype(node.input[0])) else node.get_attr_value('floating_constant_value')
        node.set_attr('value', const_val)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        node.set_attr('mode', 'constant')
        padding_before = node.get_attr_value('padding_before')
        padding_after = node.get_attr_value('padding_after')
        paddings = np.array(padding_before + padding_after).astype(np.int64)
        padding_node = ctx.make_const(utils.make_name('const'), paddings)
        node.input.append(padding_node.output[0])
        dtype = ctx.get_dtype(node.input[0])
        const_val = node.get_attr_value('integral_constant_value') if utils.is_integral_onnx_dtype(
            dtype) else node.get_attr_value('floating_constant_value')
        const_val = np.array(const_val).astype(utils.map_onnx_to_numpy_type(dtype))
        const_val_node = ctx.make_const(utils.make_name('const'), const_val)
        node.input.append(const_val_node.output[0])


@flow_op(['normalization'], flow_inputs=['x', 'gamma', 'beta', 'moving_mean', 'moving_variance'])
class BatchNorm:
    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        node.type = "BatchNormalization"
        # tf inputs: x, scale, bias, mean, variance
        # tf outputs: y, batch_mean, batch_var
        # a: data_format, epsilon, is_training
        # onnx inputs: X, scale, B, mean, variance, attributes: epsilon, momentum=0.9, spatial : 1
        # output: y, mean, var, savedmean, savedvar,
        # detach unused outputs. While we could let the unused outputs dangle,
        # some runtimes like pytorch/caffe2 do complain about it.
        if node.get_attr_value('training') or node.get_attr_value('trainable'):
            raise NotImplementedError(
                "We only support inference mode ONNX BatchNormalization now")
        consumers = [ctx.find_output_consumers(
            output_name) for output_name in node.output[1:]]
        if not any(consumers):
            new_output = [node.output[0]]
            node.output = new_output

        conv_convert_inputs(ctx, node, with_kernel=False)

        scale_shape = ctx.get_shape(node.input[1])
        mean_shape = ctx.get_shape(node.input[3])
        var_shape = ctx.get_shape(node.input[4])
        val_type = utils.map_onnx_to_numpy_type(ctx.get_dtype(node.input[1]))

        if mean_shape != scale_shape:
            new_mean_value = np.array(np.resize(node.inputs[3].get_tensor_value(as_list=False), scale_shape),
                                      dtype=val_type)
            new_mean_node_name = utils.make_name(node.name)
            ctx.make_const(new_mean_node_name, new_mean_value)
            node.input[3] = new_mean_node_name

        if var_shape != scale_shape:
            new_var_value = np.array(np.resize(node.inputs[4].get_tensor_value(as_list=False), scale_shape),
                                     dtype=val_type)
            new_val_node_name = utils.make_name(node.name)
            ctx.make_const(new_val_node_name, new_var_value)
            node.input[4] = new_val_node_name

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # is_test was removed - no change for us
        cls.version_6(ctx, node, **kwargs)


def _make_softmax_cross_entropy_with_logits(ctx, label, logit, tf_ori_node):
    label_dtype = ctx.get_dtype(label.output[0])
    logit_dtype = ctx.get_dtype(logit.output[0])
    utils.make_sure(label_dtype == logit_dtype,
                    "the following logic only works on same dtype of label and logit")

    log_softmax = ctx.make_node(op_type="LogSoftmax", inputs=logit.output)
    # implement tf.multiply(-1, tf.reduce_sum(tf.multiply(label, log_softmax), axis=1))
    mul1 = ctx.make_node(op_type="Mul", inputs=[
                         label.output[0], log_softmax.output[0]])
    reduce_sum = ctx.make_node(op_type="ReduceSum", inputs=[
                               mul1.output[0]], attr={"axes": [-1]})
    const_negative_one = ctx.make_const(name=utils.make_name("const_negative_one"),
                                        np_val=np.array(-1).astype(utils.ONNX_TO_NUMPY_DTYPE[logit_dtype]))
    mul2 = ctx.make_node(op_type="Mul", inputs=[
                         const_negative_one.output[0], reduce_sum.output[0]])
    shapes = tf_ori_node.output_shapes
    dtypes = tf_ori_node.output_dtypes
    ctx.remove_node(tf_ori_node.name)
    ctx.make_node(op_type="Squeeze", inputs=[mul2.output[0]], attr={"axes": [1]},
                  outputs=[tf_ori_node.output[0]], shapes=[shapes[0]], dtypes=[dtypes[0]])


def sparse_softmax_cross_entropy_with_logits_op_by_gathernd(ctx, node, **kwargs):
    # make subgraph to implement one_hot, idea comes from onehot_op
    indices_name = node.input[1]
    indices_shape = ctx.get_shape(indices_name)
    if len(indices_shape) != 1:
        # TODO: this works for rank=1 but tensorflow supports more than this.
        # Same principle should work but we need to implement our own eye.
        raise ValueError("onehot op: only rank1 is supported")
    logit_name = node.input[0]
    logit_dtype = ctx.get_dtype(logit_name)
    logit_shape = ctx.get_shape(logit_name)
    utils.make_sure(logit_dtype, "Dtype of {} is None".format(logit_name))
    indices_dtype = ctx.get_dtype(indices_name)
    if indices_dtype != TensorProto.INT64:
        indices_cast = ctx.make_node("Cast", [indices_name], attr={
                                     "to": TensorProto.INT64})
        indices_name = indices_cast.output[0]
    indices_size = ctx.make_node("Size", [indices_name])
    indices_unsqueeze = ctx.make_node(
        "Unsqueeze", [indices_name], attr={"axes": [1]})
    zero_const = ctx.make_const(utils.make_name(
        "zero"), np.array(0, dtype=np.int64))
    one_const = ctx.make_const(utils.make_name(
        "one"), np.array(1, dtype=np.int64))
    id_name = utils.make_name("sparse_softmax_id")
    id_output = utils.port_name(id_name)
    controlflow.make_range(ctx, zero_const.output[0], indices_size.output[0], one_const.output[0],
                           id_output, id_name, shape=[-1], dtype=TensorProto.INT64)
    id_unsqueeze = ctx.make_node("Unsqueeze", [id_output], attr={"axes": [1]})
    indices_with_id = ctx.make_node("Concat",
                                    [id_unsqueeze.output[0],
                                        indices_unsqueeze.output[0]],
                                    attr={"axis": 1})
    log_softmax = ctx.make_node(op_type="LogSoftmax",
                                inputs=[logit_name], dtypes=[logit_dtype], shapes=[logit_shape])
    gathernd_name = utils.make_name("sparse_softmax_gathernd")
    gathernd_output = utils.port_name(gathernd_name)
    tensor.make_gathernd(ctx, log_softmax.output[0], indices_with_id.output[0], gathernd_output,
                         gathernd_name, logit_dtype, [logit_shape], [logit_dtype])
    const_name = utils.make_name("const_negative_one")
    const_negative_one = ctx.make_const(
        const_name, np.array(-1).astype(utils.map_onnx_to_numpy_type(logit_dtype)))
    mul2 = ctx.make_node(op_type="Mul", inputs=[
                         const_negative_one.output[0], gathernd_output])
    shapes = node.output_shapes
    dtypes = node.output_dtypes
    ctx.remove_node(node.name)
    ctx.make_node(op_type="Squeeze",
                  inputs=[mul2.output[0]], outputs=[node.output[0]],
                  attr={"axes": [1]}, shapes=[shapes[0]], dtypes=[dtypes[0]])


@flow_op("SoftmaxCrossEntropyWithLogits")
class SoftmaxCrossEntropyWithLogits:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        logits = node.inputs[0]
        logit_dtype = ctx.get_dtype(logits.output[0])
        labels = node.inputs[1]
        label_dtype = ctx.get_dtype(labels.output[0])
        if label_dtype != logit_dtype:
            labels = ctx.make_node("Cast", labels.output, attr={
                                   "to": logit_dtype}, dtypes=[logit_dtype])

        _make_softmax_cross_entropy_with_logits(ctx, labels, logits, node)


def _make_sparse_softmax_cross_entropy_with_logits(ctx, label, logit, tf_ori_node):
    logit = logit.output[0]
    label = label.output[0]
    label_dtype = ctx.get_dtype(label)
    logit_dtype = ctx.get_dtype(logit)
    utils.make_sure(label_dtype == logit_dtype,
                    "the following logic only works on same dtype of label and logit")

    # when label is onehot, logic "tf.multiply(-1, tf.reduce_sum(tf.multiply(label, log_softmax), axis=1))" is equal to
    # "-log(q_i)" where i is the selected index specified by label, q_i = logic_i/sum, the detail process is as follows:
    # logit_exp=exp(logit) >> sum = tf.reduce_sum(logit_exp, axis = -1), masked_sum = reduce_sum(mul(logit_exp, mul))
    # >> -log(masked_sum/sum)
    logit_exp = ctx.make_node(op_type="Exp", inputs=[logit]).output[0]
    logit_exp_sum = ctx.make_node(op_type="ReduceSum", inputs=[logit_exp], attr={
                                  "axes": [-1], "keepdims": 0}).output[0]
    masked = ctx.make_node(op_type="Mul", inputs=[label, logit_exp]).output[0]
    masked_sum = ctx.make_node(op_type="ReduceSum", inputs=[masked], attr={
                               "axes": [-1], "keepdims": 0}).output[0]
    probability = ctx.make_node(op_type="Div", inputs=[
                                masked_sum, logit_exp_sum]).output[0]
    log_prob = ctx.make_node(op_type="Log", inputs=[probability]).output[0]
    const_negative_one = ctx.make_const(name=utils.make_name("const_negative_one"),
                                        np_val=np.array(-1).astype(utils.ONNX_TO_NUMPY_DTYPE[logit_dtype])).output[0]

    shapes = tf_ori_node.output_shapes
    dtypes = tf_ori_node.output_dtypes
    ctx.remove_node(tf_ori_node.name)
    res = ctx.make_node(op_type="Mul", inputs=[log_prob, const_negative_one],
                        outputs=[tf_ori_node.output[0]], shapes=[shapes[0]], dtypes=[dtypes[0]])


@flow_op("SparseSoftmaxCrossEntropyWithLogits")
class SparseSoftmaxCrossEntropyWithLogits:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # make subgraph to implement one_hot, idea comes from onehot_op
        indices_name = node.input[1]
        indices_shape = ctx.get_shape(indices_name)
        if len(indices_shape) != 1:
            # TODO: this works for rank=1 but tensorflow supports more than this.
            # Same principle should work but we need to implement our own eye.
            raise ValueError("onehot op: only rank1 is supported")
        logit_name = node.input[0]
        depth = ctx.get_shape(logit_name)[-1]
        # if number of classes is unknown or too large
        if depth == utils.ONNX_UNKNOWN_DIMENSION or depth > 20000:
            sparse_softmax_cross_entropy_with_logits_op_by_gathernd(
                ctx, node, **kwargs)
            return
        logit_dtype = ctx.get_dtype(logit_name)
        utils.make_sure(logit_dtype, "Dtype of {} is None".format(logit_name))

        dtype = utils.map_onnx_to_numpy_type(logit_dtype)
        eye = np.eye(depth).astype(dtype)
        const_name = utils.make_name("const_eye")
        const_eye = ctx.make_const(name=const_name, np_val=eye)
        onehot = ctx.make_node(op_type="Gather", inputs=[
                               const_eye.output[0], indices_name], attr={"axis": 0})
        log_softmax = ctx.make_node(op_type="LogSoftmax", inputs=[logit_name])
        # implement tf.multiply(np.float32(-1.0), tf.reduce_sum(tf.multiply(one_hot, log_softmax), axis=1))
        mul1 = ctx.make_node(op_type="Mul", inputs=[
                             onehot.output[0], log_softmax.output[0]])
        reduce_sum = ctx.make_node(op_type="ReduceSum", inputs=[
                                   mul1.output[0]], attr={"axes": [1]})
        const_name = utils.make_name("const_negative_one")
        const_negative_one = ctx.make_const(
            name=const_name, np_val=np.array(-1).astype(dtype))
        mul2 = ctx.make_node(op_type="Mul", inputs=[
                             const_negative_one.output[0], reduce_sum.output[0]])

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node(op_type="Squeeze", inputs=[mul2.output[0]], outputs=[node.output[0]], attr={"axes": [1]},
                      shapes=[shapes[0]], dtypes=[dtypes[0]])

    @classmethod
    def version_9(cls, ctx, node, **kwargs):
        # float32/64 output = SparseSoftmaxCrossEntropyWithLogits(float32/64 features, int32/64 labels)
        # the detail math process of this op is: a = onehot(labels), b = logsoftmax(features), reduce_sum(mul(a, b))
        logit_node = node.inputs[0]
        logit_shape = ctx.get_shape(node.input[0])
        logit_dtype = ctx.get_dtype(node.input[0])

        label_name = node.input[1]

        if logit_shape is not None and logit_shape[-1] != -1:
            num_class = logit_shape[-1]
            node_nme = utils.make_name("onehot_depth")
            depth_node = ctx.make_const(node_nme, np.array(
                [num_class]).astype(np.int64)).output[0]
        else:
            logit_shape = ctx.make_node("Shape", [node.input[0]]).output[0]
            slice_args = {"data": logit_shape,
                          "starts": [-1], "ends": [int(utils.get_max_value(np.int32))]}
            num_class = GraphBuilder(ctx).make_slice(kwargs=slice_args)
            depth_node = num_class
        values_node = ctx.make_const(utils.make_name(
            "onehot_values"), np.array([0, 1]).astype(np.int64)).output[0]
        label_dtype = ctx.get_dtype(label_name)
        if label_dtype != TensorProto.INT64:
            onehot_indice = ctx.make_node("Cast", [label_name], attr={
                                          "to": TensorProto.INT64}).output[0]
        else:
            onehot_indice = label_name
        label_node = ctx.make_node(op_type="OneHot",
                                   inputs=[onehot_indice, depth_node, values_node])
        # the above logic makes output dtype of label_node now always int64
        # make sure label has same dtype as logit
        if logit_dtype != TensorProto.INT64:
            label_node = ctx.make_node("Cast", label_node.output, attr={
                                       "to": logit_dtype}, dtypes=[logit_dtype])

        _make_sparse_softmax_cross_entropy_with_logits(
            ctx, label_node, logit_node, node)

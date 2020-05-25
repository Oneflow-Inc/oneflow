# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
reduction
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from onnx import onnx_pb, helper

from oneflow.python.onnx import utils
from oneflow.python.onnx.handler import tf_op

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring

@tf_op("Min", onnx_op="ReduceMin")
@tf_op("Max", onnx_op="ReduceMax")
@tf_op("Mean", onnx_op="ReduceMean")
@tf_op("Sum", onnx_op="ReduceSum")
@tf_op("Prod", onnx_op="ReduceProd")
class ReduceOpBase:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        axes_node = node.inputs[1]
        axes = axes_node.get_tensor_value()
        if np.isscalar(axes):
            axes = [axes]
        input_shape = ctx.get_shape(node.input[0])
        if input_shape is None:
            if any([val < 0 for val in axes]):
                raise ValueError("reduce_op: cannot have negative axis because we don't know input rank")
        else:
            input_rank = len(ctx.get_shape(node.input[0]))
            axes = [val + input_rank if val < 0 else val for val in axes]

        node.set_attr("axes", axes)
        ctx.remove_input(node, node.input[1])
        keep_dims = node.get_attr("keep_dims")
        if keep_dims:
            del node.attr['keep_dims']
            node.set_attr("keepdims", keep_dims.i)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)


@tf_op(["ArgMax", "ArgMin"])
class ArgMax:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # output_type output = ArgMin(T input, Tidx dimension, @type Tidx, @type output_type)
        # tensor(int32) reduced = ArgMin(T data, @INT axis, @INT keepdims)
        axis_node = node.inputs[1]
        axis = axis_node.get_tensor_value()
        if axis < 0:
            # ArgMax|ArgMin in onnx don't necessary support negative axis(not in doc explicitly)
            input_shape = ctx.get_shape(node.input[0])
            dim_count = len(input_shape) if input_shape else 0
            axis = dim_count + axis

        # TF ArgMin/ArgMax may return int32 or int64
        # Onnx ArgMin/ArgMax only supports int64 output, add cast if needed
        if node.get_attr_int("output_type") == onnx_pb.TensorProto.INT32:
            # current node will return int64 after conversion, which differs from previous dtype got from tf
            ctx.set_dtype(node.output[0], onnx_pb.TensorProto.INT64)
            op_name = utils.make_name("Cast")
            cast_node = ctx.insert_new_node_on_output("Cast", node.output[0], name=op_name,
                                                      to=onnx_pb.TensorProto.INT32)
            ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT32)
            ctx.copy_shape(node.output[0], cast_node.output[0])

        node.set_attr("axis", axis)
        node.set_attr("keepdims", 0)
        ctx.remove_input(node, node.input[1])

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic same
        cls.version_1(ctx, node, **kwargs)


@tf_op(["All", "Any"])
class AllAny:
    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        # T output = All(T x, list(int) reduce_indices, @bool keepdims)
        # T output = Any(T x, list(int) reduce_indices, @bool keepdims)
        reduce_dim = node.inputs[1].get_tensor_value()

        # for Any, the reduce_indices can be scalar as observed.
        if np.isscalar(reduce_dim):
            reduce_dim = [reduce_dim]

        utils.make_sure(all(i >= 0 for i in reduce_dim), "negative reduce axis is not supported in onnx for now")

        cast = ctx.make_node(op_type="Cast", inputs=[node.input[0]], attr={"to": onnx_pb.TensorProto.FLOAT})
        keepdims = helper.get_attribute_value(node.get_attr("keep_dims"))
        op_type = "ReduceMin" if node.type == "All" else "ReduceSum"
        reduce_node = ctx.make_node(op_type=op_type, inputs=cast.output,
                                    attr={"axes": reduce_dim, "keepdims": keepdims})

        zero_node = ctx.make_const(utils.make_name("zero_reduce"), np.array(0, dtype=np.float32))

        shapes = node.output_shapes
        dtypes = node.output_dtypes
        ctx.remove_node(node.name)
        ctx.make_node(op_type="Greater", inputs=[reduce_node.output[0], zero_node.output[0]],
                      name=node.name, outputs=node.output, shapes=shapes, dtypes=dtypes)


@tf_op("AddN")
class AddN():
    @classmethod
    def version_6(cls, ctx, node, **kwargs):
        node.type = "Sum"

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

from oneflow.python.onnx import util
from oneflow.python.onnx.handler import flow_op

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring


@flow_op("reduce_min", onnx_op="ReduceMin")
# reduce_max is not user op
# @flow_op("reduce_max", onnx_op="ReduceMax")
@flow_op("reduce_sum", onnx_op="ReduceSum")
@flow_op("reduce_prod", onnx_op="ReduceProd")
class ReduceOpBase:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        axes = node.get_attr_value("axis")
        input_shape = ctx.get_shape(node.input[0])
        if input_shape is None:
            if any([val < 0 for val in axes]):
                raise ValueError(
                    "reduce_op: cannot have negative axis because we don't know input rank"
                )
        else:
            input_rank = len(ctx.get_shape(node.input[0]))
            axes = [val + input_rank if val < 0 else val for val in axes]

        # axes == [] means reducing all axes, which is also the default value of onnx
        if len(axes) > 0:
            node.set_attr("axes", axes)
        keep_dims = node.get_attr("keepdims")
        if keep_dims:
            del node.attr["keepdims"]
            node.set_attr("keepdims", keep_dims.i)

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.version_1(ctx, node, **kwargs)


@flow_op(["argmax", "argmin"], ["ArgMax", "ArgMin"])
class ArgMax:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        # output_type output = ArgMin(T input, Tidx dimension, @type Tidx, @type output_type)
        # tensor(int32) reduced = ArgMin(T data, @INT axis, @INT keepdims)
        input_shape = ctx.get_shape(node.input[0])
        dim_count = len(input_shape) if input_shape else 0
        axis = dim_count - 1

        # Onnx ArgMin/ArgMax only supports int64 output, add cast if needed
        if ctx.get_dtype(node.output[0]) == onnx_pb.TensorProto.INT32:
            # current node will return int64 after conversion, which differs from previous dtype got from oneflow
            ctx.set_dtype(node.output[0], onnx_pb.TensorProto.INT64)
            op_name = util.make_name("Cast")
            cast_node = ctx.insert_new_node_on_output(
                "Cast", node.output[0], name=op_name, to=onnx_pb.TensorProto.INT32
            )
            ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT32)
            ctx.copy_shape(node.output[0], cast_node.output[0])

        node.set_attr("axis", axis)
        node.set_attr("keepdims", 0)
        ctx.remove_input(node, node.input[1])

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic same
        cls.version_1(ctx, node, **kwargs)

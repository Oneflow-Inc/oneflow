# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
logical
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from onnx import TensorProto
from oneflow.python.framework import id_util
from oneflow.python.onnx import util
from oneflow.python.onnx.handler import flow_op
from oneflow.python.onnx.onnx_opset import common


logger = logging.getLogger(__name__)

# pylint: disable=unused-argument,missing-docstring


def _add_cast_to_inputs(graph, node, supported_dtypes, target_dtype):
    is_support = True
    for inp in node.input:
        if graph.get_dtype(inp) not in supported_dtypes:
            is_support = False
            break
    if not is_support:
        for inp in node.input:
            inp_cast = graph.insert_new_node_on_input(
                node, "Cast", inp, to=target_dtype
            )
            graph.copy_shape(inp, inp_cast.output[0])
            graph.set_dtype(inp_cast.output[0], target_dtype)


def _add_cast_to_output(graph, node):
    # oneflow logical ops produce int8 tensor while onnx logical ops produce bool tensor
    output = node.output[0]
    cast_node = graph.insert_new_node_on_output(
        "Cast", output, id_util.UniqueStr("cast"), to=graph.get_dtype(output)
    )
    graph.copy_shape(output, node.output[0])
    graph.set_dtype(node.output[0], TensorProto.BOOL)


# oneflow doesn't have logical_not and broadcast_logical_or, but
# it is easy to implement onnx converter in advance
@flow_op("logical_not", onnx_op="Not")
class DirectOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        _add_cast_to_output(ctx, node)


@flow_op("broadcast_logical_and", onnx_op="And", flow_inputs=["x", "y"])
@flow_op("broadcast_logical_or", onnx_op="Or", flow_inputs=["x", "y"])
class BroadcastOp(common.BroadcastOp):
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        _add_cast_to_output(ctx, node)
        super().version_1(ctx, node, **kwargs)


@flow_op(
    ["broadcast_equal", "broadcast_not_equal"],
    ["Equal", "NotEqual"],
    flow_inputs=["x", "y"],
)
class Equal:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        _add_cast_to_output(ctx, node)
        need_not = node.type == "NotEqual"
        common.BroadcastOp.version_1(ctx, node, **kwargs)
        if need_not:
            node.type = "Equal"
            output_name = node.output[0]
            not_node = ctx.insert_new_node_on_output(
                "Not", output_name, name=id_util.UniqueStr(node.name)
            )
            ctx.copy_shape(output_name, not_node.output[0])
            ctx.copy_dtype(output_name, not_node.output[0])

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        # T2 output = Equal(T1, x, T1 y), T1 \in {bool, int32, int64}
        _add_cast_to_output(ctx, node)
        need_not = node.type == "NotEqual"
        supported_dtypes = [TensorProto.BOOL, TensorProto.INT32, TensorProto.INT64]
        # FIXME: casting is not the same as equal
        target_dtype = TensorProto.INT32
        _add_cast_to_inputs(ctx, node, supported_dtypes, target_dtype)
        if need_not:
            node.type = "Equal"
            output_name = node.output[0]
            not_node = ctx.insert_new_node_on_output(
                "Not", output_name, name=id_util.UniqueStr(node.name)
            )
            ctx.copy_shape(output_name, not_node.output[0])
            ctx.copy_dtype(output_name, not_node.output[0])

    @classmethod
    def version_11(cls, ctx, node, **kwargs):
        # starting with opset-11, equal supports all types
        _add_cast_to_output(ctx, node)
        need_not = node.type == "NotEqual"
        if need_not:
            node.type = "Equal"
            output_name = node.output[0]
            not_node = ctx.insert_new_node_on_output(
                "Not", output_name, name=id_util.UniqueStr(node.name)
            )
            ctx.copy_shape(output_name, not_node.output[0])
            ctx.copy_dtype(output_name, not_node.output[0])


@flow_op(
    ["broadcast_greater", "broadcast_less"], ["Greater", "Less"], flow_inputs=["x", "y"]
)
class GreaterLess:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        _add_cast_to_output(ctx, node)
        common.BroadcastOp.version_1(ctx, node, **kwargs)

    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        _add_cast_to_output(ctx, node)
        # T2 output = Greater(T1 x, T1 y), T2=tensor(bool)
        # T2 output = Less(T1 x, T1 y), T2=tensor(bool)
        # Great/Less in opset7 only supports limited types, insert Cast if needed
        supported_dtypes = [TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE]
        target_dtype = TensorProto.FLOAT
        _add_cast_to_inputs(ctx, node, supported_dtypes, target_dtype)


@flow_op("broadcast_greater_equal", onnx_op="Less", flow_inputs=["x", "y"])
@flow_op("broadcast_less_equal", onnx_op="Greater", flow_inputs=["x", "y"])
class GreaterLessEqual:
    @classmethod
    def version_7(cls, ctx, node, **kwargs):
        _add_cast_to_output(ctx, node)
        GreaterLess.version_7(ctx, node, **kwargs)
        output_name = node.output[0]
        new_node = ctx.insert_new_node_on_output(
            "Not", output_name, name=id_util.UniqueStr(node.name)
        )
        ctx.copy_shape(output_name, new_node.output[0])
        ctx.set_dtype(new_node.output[0], ctx.get_dtype(output_name))

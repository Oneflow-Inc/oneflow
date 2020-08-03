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
import sys

import numpy as np
from onnx import numpy_helper
from onnx import onnx_pb
from onnx.onnx_pb import TensorProto

import oneflow.python.onnx
from oneflow.python.framework import id_util
from oneflow.python.onnx import constants, util
from oneflow.python.onnx.graph_builder import GraphBuilder
from oneflow.python.onnx.handler import flow_op
from oneflow.python.onnx.onnx_opset import nn, math

logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement


def _ConvertShapeNodeToInt64(ctx, node, input_number):
    """cast int32 shape into int64 shape."""
    name = node.input[input_number]

    cast_node = ctx.InsertNewNodeOnInput(node, "Cast", name)
    cast_node.set_attr("to", onnx_pb.TensorProto.INT64)
    ctx.set_dtype(cast_node.output[0], onnx_pb.TensorProto.INT64)
    ctx.CopyShape(name, cast_node.output[0])


def _WrapConcatWithCast(ctx, node):
    """wrap concat in casts for opset < 8 since it only supports."""
    supported_types = [onnx_pb.TensorProto.FLOAT, onnx_pb.TensorProto.FLOAT16]
    dtype = ctx.get_dtype(node.output[0])
    need_casting = dtype not in supported_types
    if need_casting:
        output_name = node.output[0]
        # cast each inputs to float
        for i, inp in enumerate(node.inputs):
            input_cast = ctx.InsertNewNodeOnInput(node, "Cast", node.input[i])
            input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
            ctx.set_dtype(input_cast.output[0], onnx_pb.TensorProto.FLOAT)
        next_nodes = ctx.FindOutputConsumers(node.output[0])
        # cast output back to dtype unless the next op is a cast
        if next_nodes[0].type != "Cast":
            op_name = id_util.UniqueStr(node.name)
            output_cast = ctx.InsertNewNodeOnOutput("Cast", output_name, name=op_name)
            output_cast.set_attr("to", dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.CopyShape(output_name, output_cast.output[0])


@flow_op("reshape", "Reshape")
class Reshape:
    @classmethod
    def Version_5(cls, ctx, node, **kwargs):
        dtype = ctx.get_dtype(node.output[0])
        need_casting = dtype in [
            onnx_pb.TensorProto.INT32,
            onnx_pb.TensorProto.INT16,
            onnx_pb.TensorProto.INT64,
        ]
        shape_node = ctx.MakeConst(
            id_util.UniqueStr("shape"), np.array(node.get_attr_value("shape"))
        )
        node.input = node.input + [shape_node.name]
        if ctx.opset >= 8 or not need_casting:
            # onnx reshape can handle the type - done
            return

        # onnx < opset 8 does not know reshape for other types than float*, wrap the reshape in casts
        input_cast = ctx.InsertNewNodeOnInput(node, "Cast", node.input[0])
        input_cast.set_attr("to", onnx_pb.TensorProto.FLOAT)
        ctx.CopyShape(node.output[0], input_cast.output[0])

        # if the next node is already a cast we don't need to insert another one
        next_nodes = ctx.FindOutputConsumers(node.output[0])
        if len(next_nodes) != 1 or next_nodes[0].type != "Cast":
            op_name = id_util.UniqueStr(node.name)
            output_cast = ctx.InsertNewNodeOnOutput(
                "Cast", node.output[0], name=op_name
            )
            output_cast.set_attr("to", dtype)
            ctx.set_dtype(output_cast.output[0], dtype)
            ctx.CopyShape(node.output[0], output_cast.output[0])


@flow_op("squeeze", "Squeeze")
class Squeeze:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # T output = Squeeze(T input, @list(int) squeeze_dims)
        # T squeezed = Squeeze(T data, @AttrType.INTS axes), axes are list of positive integers.
        axis = node.get_attr_value("axes")

        neg_axis = any([val < 0 for val in axis])
        if neg_axis:
            shape = ctx.get_shape(node.input[0])
            util.MakeSure(shape is not None, "squeeze input shape cannot be None")
            shape_len = len(shape)
            axis = [a + shape_len if a < 0 else a for a in axis]
        node.set_attr("axes", axis)

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.Version_1(ctx, node, **kwargs)


@flow_op("transpose", onnx_op="Transpose")
class Transpose:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # T y = Transpose(T x, Tperm perm, @type Tperm)
        # T transposed = Transpose(T data, @INTS perm)
        if len(node.input) > 1:
            perm = node.inputs[1]
            if perm.is_const():
                # perms is passed as const
                dims = perm.get_tensor_value()
                ctx.RemoveInput(node, node.input[1])
                node.set_attr("perm", dims)
            else:
                util.MakeSure(False, "perm can't be dynamic in ONNX")
        else:
            # graph rewrite moved perm to attribute
            pass


@flow_op("concat", "Concat")
class Concat:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # old concat op has axis as input[0]
        axis_val = node.get_attr_value("axis")

        if axis_val < 0:
            input_shape = ctx.get_shape(node.input[0])
            axis_val = len(input_shape) + axis_val
        node.set_attr("axis", axis_val)

        if ctx.opset < 8:
            # opset < 8: might need to wrap concat in casts since only float is supported
            _WrapConcatWithCast(ctx, node)
            return

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.Version_1(ctx, node, **kwargs)


@flow_op("gather_nd", onnx_op="GatherND", flow_ibns=["params", "indices"])
class GatherND:
    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # indicies input
        input1 = node.input[1]
        target_dtype = TensorProto.INT64
        if ctx.get_dtype(input1) != TensorProto.INT64:
            inp_cast = ctx.InsertNewNodeOnInput(node, "Cast", input1, to=target_dtype)
            ctx.CopyShape(input1, inp_cast.output[0])
            ctx.set_dtype(inp_cast.output[0], target_dtype)


@flow_op("cast", "Cast")
class Cast:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # DstT y = Cast(SrcT x, @type SrcT, @type DstT)
        # T2 output = Cast(T1 input, @STRING to)
        dst = node.get_attr("dtype")
        dst = oneflow.python.onnx.util.ONNX_DTYPE_NAMES[dst]
        node.set_attr("to", dst)

    @classmethod
    def Version_6(cls, ctx, node, **kwargs):
        dst = node.get_attr_value("dtype")
        node.set_attr("to", dst)

    @classmethod
    def Version_9(cls, ctx, node, **kwargs):
        cls.Version_6(ctx, node, **kwargs)


@flow_op("identity", "Identity")
class Identity:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        pass

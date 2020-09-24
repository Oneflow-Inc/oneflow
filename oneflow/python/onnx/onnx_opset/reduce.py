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
from onnx import onnx_pb, helper

from oneflow.python.framework import id_util
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
    def Version_1(cls, ctx, node, **kwargs):
        axes = node.attrs.get("axis", None)
        input_shape = ctx.get_shape(node.input_tensor_names[0])
        if input_shape is None:
            if any([val < 0 for val in axes]):
                raise ValueError(
                    "reduce_op: cannot have negative axis because we don't know input rank"
                )
        else:
            input_rank = len(ctx.get_shape(node.input_tensor_names[0]))
            axes = [val + input_rank if val < 0 else val for val in axes]

        # axes == [] means reducing all axes, which is also the default value of onnx
        if len(axes) > 0:
            node.attrs["axes"] = axes

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic is same
        cls.Version_1(ctx, node, **kwargs)


@flow_op(["argmax", "argmin"], ["ArgMax", "ArgMin"])
class ArgMax:
    @classmethod
    def Version_1(cls, ctx, node, **kwargs):
        # output_type output = ArgMin(T input, Tidx dimension, @type Tidx, @type output_type)
        # tensor(int32) reduced = ArgMin(T data, @INT axis, @INT keepdims)
        input_shape = ctx.get_shape(node.input_tensor_names[0])
        dim_count = len(input_shape) if input_shape else 0
        axis = dim_count - 1

        # Onnx ArgMin/ArgMax only supports int64 output, add cast if needed
        if ctx.get_dtype(node.output_tensor_names[0]) == onnx_pb.TensorProto.INT32:
            # current node will return int64 after conversion, which differs from previous dtype got from oneflow
            ctx.set_dtype(node.output_tensor_names[0], onnx_pb.TensorProto.INT64)
            op_name = id_util.UniqueStr("Cast")
            cast_node = ctx.InsertNewNodeOnOutput(
                "Cast",
                node.output_tensor_names[0],
                name=op_name,
                to=onnx_pb.TensorProto.INT32,
            )
            ctx.set_dtype(cast_node.output_tensor_names[0], onnx_pb.TensorProto.INT32)
            ctx.CopyShape(node.output_tensor_names[0], cast_node.output_tensor_names[0])

        node.attrs["axis"] = axis
        node.attrs["keepdims"] = 0
        ctx.RemoveInput(node, node.input_tensor_names[1])

    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # Opset 11 supports negative axis, but core logic same
        cls.Version_1(ctx, node, **kwargs)

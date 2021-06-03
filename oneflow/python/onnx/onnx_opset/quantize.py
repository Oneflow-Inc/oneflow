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
from typing import Optional, Callable
from oneflow.python.framework import id_util
from oneflow.python.onnx.graph import Graph, Node
from oneflow.python.onnx.handler import flow_op


logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement


@flow_op("min_max_observer", flow_obns=["scale", "zero_point"])
class MinMaxObserver:
    @classmethod
    def _Convert(cls, ctx: Graph, node: Node, opset: int, **kwargs):
        bit = node.attrs["quantization_bit"]
        scheme = node.attrs["quantization_scheme"]
        per_layer = node.attrs["per_layer_quantization"]
        formula = node.attrs["quantization_formula"]

        if not per_layer and opset == 10:
            raise NotImplementedError("per-channel mode is not supported in version 10")

        input_node: Node = node.input_nodes[0]
        input_np: np.ndarray = input_node.get_tensor_value(as_list=False)

        input_np = (
            input_np.flatten()
            if formula == "cambricon" or per_layer
            else input_np.reshape((input_np.shape[0], -1))
        )

        def get_min_or_max_value(get_min: bool, pre_func: Optional[Callable] = None):
            data = input_np.copy()
            func = np.min if get_min else np.max
            if pre_func is not None:
                data = pre_func(data)
            result = func(data, axis=-1 if formula == "cambricon" or per_layer else 1)
            return result.flatten()

        input_np_abs_max = get_min_or_max_value(False, np.abs)

        if formula == "google":
            if scheme == "symmetric":
                denominator = 2.0 ** (bit - 1) - 1
                scale = input_np_abs_max / denominator
                zero_point = np.array([0] * scale.shape[0], dtype=np.int8)

            elif scheme == "affine":
                input_np_min = get_min_or_max_value(True)
                denominator = 2.0 ** bit - 1
                scale = (get_min_or_max_value(False) - input_np_min) / denominator
                zero_point = (-np.round(input_np_min / scale)).astype(np.uint8)

            else:
                raise ValueError("invalid quantization scheme: " + scheme)

        elif formula == "cambricon":
            scale = np.floor(np.log2(input_np_abs_max)) - (bit - 2)
            zero_point = np.array([0], dtype=np.int8)

        else:
            raise ValueError("invalid quantization formula: " + formula)

        ctx.RemoveNode(node.name)
        ctx.MakeConst(node.output_tensor_names[0], scale)
        ctx.MakeConst(node.output_tensor_names[1], zero_point)

    @classmethod
    def Version_10(cls, ctx: Graph, node: Node, **kwargs):
        cls._Convert(ctx, node, opset=10, **kwargs)

    @classmethod
    def Version_13(cls, ctx: Graph, node: Node, **kwargs):
        cls._Convert(ctx, node, opset=13, **kwargs)


@flow_op(
    "moving_average_min_max_observer",
    flow_ibns=["in", "current_train_step", "moving_max", "moving_min"],
    flow_obns=["scale", "zero_point"],
)
class MovingAverageMinMaxObserver:
    @classmethod
    def Version_10(cls, ctx: Graph, node: Node, **kwargs):
        bit = node.attrs["quantization_bit"]
        scheme = node.attrs["quantization_scheme"]
        formula = node.attrs["quantization_formula"]

        moving_max_node: Node = node.input_nodes[2]
        moving_max_np: np.ndarray = moving_max_node.get_tensor_value(as_list=False)
        moving_min_node: Node = node.input_nodes[3]
        moving_min_np: np.ndarray = moving_min_node.get_tensor_value(as_list=False)

        _zero = np.array([0], dtype=np.int8)

        if formula == "google":
            if scheme == "symmetric":
                denominator = 2.0 ** (bit - 1) - 1
                scale = moving_max_np / denominator
                zero_point = _zero

            elif scheme == "affine":
                denominator = 2.0 ** bit - 1
                scale = (moving_max_np - moving_min_np) / denominator
                zero_point = (
                    (-np.round(moving_min_np / scale)).astype(np.uint8).flatten()
                )
            else:
                raise ValueError("invalid quantization scheme: " + scheme)

        elif formula == "cambricon":
            scale = np.floor(np.log2(moving_max_np)) - (bit - 2)
            zero_point = _zero

        else:
            raise ValueError("invalid quantization formula: " + formula)

        ctx.RemoveNode(node.name)
        ctx.MakeConst(node.output_tensor_names[0], scale.flatten())
        ctx.MakeConst(node.output_tensor_names[1], zero_point)


@flow_op(
    "fake_quantization",
    onnx_op="QuantizeLinear",
    flow_ibns=["in", "scale", "zero_point"],
)
class FakeQuantization:
    @classmethod
    def _Convert(cls, ctx: Graph, node: Node, opset: int, **kwargs):
        formula = node.attrs["quantization_formula"]
        if formula == "cambricon":
            raise ValueError("invalid quantization formula: " + formula)

        dequant_node = ctx.InsertNewNodeOnOutput(
            "DequantizeLinear",
            node.output_tensor_names[0],
            name=id_util.UniqueStr(node.name),
        )
        if opset < 13:
            scale_shape = ctx.get_shape(node.input_tensor_names[1])
            if not (len(scale_shape) == 1 and scale_shape[0] == 1):
                raise RuntimeError("per-channel mode is not supported in version 10")

        else:
            node.attrs["axis"] = 0
            dequant_node.attrs["axis"] = 0

        dequant_node.input_tensor_names = [
            node.output_tensor_names[0],
            node.input_tensor_names[1],
            node.input_tensor_names[2],
        ]

        ctx.set_dtype(
            dequant_node.output_tensor_names[0],
            ctx.get_dtype(node.input_tensor_names[0]),
        )
        ctx.CopyShape(node.output_tensor_names[0], dequant_node.output_tensor_names[0])

    @classmethod
    def Version_10(cls, ctx: Graph, node: Node, **kwargs):
        cls._Convert(ctx, node, 10, **kwargs)

    @classmethod
    def Version_13(cls, ctx: Graph, node: Node, **kwargs):
        cls._Convert(ctx, node, 13, **kwargs)

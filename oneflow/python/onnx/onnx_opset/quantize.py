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
from onnx.onnx_pb import TensorProto
from oneflow.python.onnx.graph import Graph, Node
from oneflow.python.onnx.handler import flow_op


logger = logging.getLogger(__name__)


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement


@flow_op("min_max_observer", flow_obns=["scale", "zero_point"])
class MinMaxObserver:
    @classmethod
    def Version_11(cls, ctx: Graph, node: Node, **kwargs):
        bit = node.attrs["quantization_bit"]
        scheme = node.attrs["quantization_scheme"]
        per_layer = node.attrs["per_layer_quantization"]
        formula = node.attrs["quantization_formula"]

        input_node: Node = node.input_nodes[0]
        input_np: np.ndarray = input_node.get_tensor_value(as_list=False)

        _input_np = (
            input_np.flatten()
            if per_layer
            else input_np.reshape((input_np.shape[0], -1))
        )
        _input_np_abs_max = np.max(np.abs(_input_np), axis=-1 if per_layer else 1)

        if formula == "google":
            if scheme == "symmetric":
                denominator = 2.0 ** (bit - 1) - 1
                scale = _input_np_abs_max / denominator
                zero_point = (
                    np.array([0], dtype=np.int8)
                    if per_layer
                    else np.array([0 for _ in range(scale.shape[0])], dtype=np.int8)
                )

            if scheme == "affine":
                _input_np_min = _input_np.min(axis=-1 if per_layer else 1)
                denominator = 2.0 ** bit - 1
                scale = (
                    _input_np.max(axis=-1 if per_layer else 1) - _input_np_min
                ) / denominator
                zero_point = -_input_np_min / scale
                zero_point = (
                    np.array([zero_point], dtype=np.uint8)
                    if per_layer
                    else zero_point.astype(np.uint8)
                )

        if formula == "cambricon":
            scale = np.floor(np.log2(_input_np_abs_max)) - (bit - 2)
            zero_point = np.array([0], dtype=np.int8)

        ctx.RemoveNode(node.name)
        ctx.MakeConst(node.output_tensor_names[0], scale.flatten())
        ctx.MakeConst(node.output_tensor_names[1], zero_point.flatten())
        ctx.set_dtype(node.output_tensor_names[1], TensorProto.UINT8)


@flow_op(
    "moving_average_min_max_observer",
    flow_ibns=["in", "current_train_step", "moving_max", "moving_min"],
    flow_obns=["scale", "zero_point"],
)
class MovingAverageMinMaxObserver:
    @classmethod
    def Version_11(cls, ctx: Graph, node: Node, **kwargs):
        bit = node.attrs["quantization_bit"]
        scheme = node.attrs["quantization_scheme"]
        formula = node.attrs["quantization_formula"]

        moving_max_node: Node = node.input_nodes[2]
        moving_max_np: np.ndarray = moving_max_node.get_tensor_value(as_list=False)
        moving_min_node: Node = node.input_nodes[3]
        moving_min_np: np.ndarray = moving_min_node.get_tensor_value(as_list=False)
        if formula == "google":
            if scheme == "symmetric":
                denominator = 2.0 ** (bit - 1) - 1
                scale = moving_max_np / denominator
                zero_point = 0
            if scheme == "affine":
                denominator = 2.0 ** bit - 1
                scale = (moving_max_np - moving_min_np) / denominator
                zero_point = -moving_min_np / scale

        if formula == "cambricon":
            scale = np.floor(np.log2(moving_max_np)) - (bit - 2)
            zero_point = 0

        scale = np.array([scale])
        zero_point = np.array([zero_point], dtype=np.int8)
        if formula == "google" and scheme == "affine":
            zero_point = zero_point.astype(np.uint8)

        ctx.RemoveNode(node.name)
        ctx.MakeConst(node.output_tensor_names[0], scale.flatten())
        ctx.MakeConst(node.output_tensor_names[1], zero_point.flatten())

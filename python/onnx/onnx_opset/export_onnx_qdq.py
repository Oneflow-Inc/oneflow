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
from onnx import TensorProto
from oneflow.python.framework import id_util
from oneflow.python.onnx import constants, util
from oneflow.python.onnx.handler import flow_op
from oneflow.python.onnx.onnx_opset import common
@flow_op("min_max_observer", onnx_op="")
class min_max_observer:
    @classmethod
    def Version_11(cls, ctx, node, **kwargs):

        # step1: 根据 min_max_observer op 的 input 和 attrs 手动计算 scale 和 zero_point
        weight_node = node.input_tensor_names[0]
        weight = weight_node.get_tensor_value()
        weight_np = weight.numpy()
        weight_flatten = weight_np.flatten()

        quantization_bit = node.attrs["quantization_bit"]
        quantization_scheme = node.attrs["quantization_scheme"]

        if quantization_scheme == "symmetric":
            weight_max = np.max(np.abs(weight_flatten)) # weight 需要转成 numpy?
            denominator = 2.0 ** (quantization_bit - 1) -1

            scale_np = weight_max / denominator
            zero_point_np = 0

            # step2: 删除原 min_max_observer op 的 node
            ctx.RemoveNode(node.name)

            # step3：新建两个表示 scale 和 zero_point 的 const op
            scale_node = ctx.MakeConst(id_util.UniqueStr("scale"), scale_np)
            zero_point_node = ctx.MakeConst(id_util.UniqueStr("zero_point"), zero_point_np)
        else:
            weight_max = np.max(weight_flatten)
            weight_min = np.min(weight_flatten)
            denominator = 2.0 ** (quantization_bit) - 1

            scale_np = (weight_max - weight_min) / denominator
            zero_point_np = -weight_min / scale_np

            # step2: 删除原 min_max_observer op 的 node
            ctx.RemoveNode(node.name)

            # step3：新建两个表示 scale 和 zero_point 的 const op
            scale_node = ctx.MakeConst(id_util.UniqueStr("scale"), scale_np)
            zero_point_node = ctx.MakeConst(id_util.UniqueStr("zero_point"), zero_point_np)

@flow_op("moving_average_min_maxObserver", onnx_op="")
class moving_average_min_maxObserver:
    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # step1: 根据 min_max_observer op 的 input 和 attrs 手动计算 scale 和 zero_point
        input_node = node.input_tensor_names[0]
        moving_max_node = node.input_tensor_names[2]
        moving_min_node = node.input_tensor_names[3]

        activation = input_node.get_tensor_value()
        activation_np = activation.numpy()

        moving_max = moving_max_node.get_tensor_value()
        moving_max_np = moving_max.numpy()

        moving_min = moving_min_node.get_tensor_value()
        moving_min_np = moving_min.numpy()
        
        quantization_bit = node.attrs["quantization_bit"]
        quantization_scheme = node.attrs["quantization_scheme"]
        momentum = node.attrs["momentum"]

        if quantization_scheme == "symmetric":
            activation_np_max = np.max(np.abs(activation_np))
            denominator = 2.0 ** (quantization_bit -1) -1

            if moving_max_np[0] == 0:
                moving_max_np[0] = activation_np_max
            else:
                moving_max_np[0] = moving_max_np[0] * momentum + activation_np_max * (1 - momentum)

            scale_np = moving_max_np[0] / denominator
            zero_point_np = 0

            # step2: 删除原 min_max_observer op 的 node
            ctx.RemoveNode(node.name)

            # step3：新建两个表示 scale 和 zero_point 的 const op
            scale_node = ctx.MakeConst(id_util.UniqueStr("scale"), scale_np)
            zero_point_node = ctx.MakeConst(id_util.UniqueStr("zero_point"), zero_point_np)

        else: # affine
            activation_np_max = np.max(activation_np)
            activation_np_min = np.min(activation_np)

            denominator = 2.0 ** (quantization_bit) - 1

            if moving_max_np[0] == 0:
                moving_max_np[0] = activation_np_max
            else:
                moving_max_np[0] = moving_max_np[0] * momentum + activation_np_max * (1 - momentum)

            if moving_min_np[0] == 0:
                moving_min_np[0] = activation_np_min
            else:
                moving_min_np[0] = moving_min_np[0] * momentum + activation_np_min * (1 - momentum)

            scale_np = (moving_max_np[0] - moving_min_np[0]) / denominator
            zero_point_np = -moving_min_np[0] / scale_np

            # step2: 删除原 min_max_observer op 的 node
            ctx.RemoveNode(node.name)

            # step3：新建两个表示 scale 和 zero_point 的 const op
            scale_node = ctx.MakeConst(id_util.UniqueStr("scale"), scale_np)
            zero_point_node = ctx.MakeConst(id_util.UniqueStr("zero_point"), zero_point_np)
            

@flow_op("fake_quantization", onnx_op="QuantizeLinear", flow_ibns=["in", "scale", "zero_point"])
class Fake_Quantization():
    @classmethod
    def Version_11(cls, ctx, node, **kwargs):
        # 步骤1：得到 scale / zero_point
        input_node = node.input_tensor_names[0]
        scale_node = node.input_tensor_names[1]
        zero_point_node = node.input_tensor_names[2]
        input = input_node.get_tensor_value()
        scale = scale_node.get_tensor_value()
        zero_point = zero_point_node.get_tensor_value()
        quantization_bit = node.attrs["quantization_bit"] 
        quantization_scheme = node.attrs["quantization_scheme"]

        # 插入 DequantizeLinear op
        output_name = node.output_tensor_names[0]
        dequant_node = ctx.InsertNewNodeOnOutput(
            "DequantizeLinear", output_name, name=id_util.UniqueStr(node.name)
        )
        ctx.set_dtype(dequant_node.output_tensor_names[0], ctx.get_dtype(node.output_tensor_names[0]))
        ctx.CopyShape(node.output_tensor_names[0], dequant_node.output_tensor_names[0])
        actual_outputs = dequant_node.output_tensor_names
        
       
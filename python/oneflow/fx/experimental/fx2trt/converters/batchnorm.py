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
import torch
import numpy as np
import tensorrt as trt
from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter

from .helper_functions import mark_as_int8_layer, to_numpy, get_dyn_range


def common_batchnorm(network, mod, input_val, layer_name, is_quantized):
    scale = to_numpy(mod.weight) / np.sqrt(to_numpy(mod.running_var) + mod.eps)
    bias = to_numpy(mod.bias) - to_numpy(mod.running_mean) * scale
    power = np.ones_like(scale)

    layer = network.add_scale(input_val, trt.ScaleMode.CHANNEL, bias, scale, power)
    layer.name = layer_name

    if is_quantized:
        mark_as_int8_layer(
            layer, get_dyn_range(mod.scale, mod.zero_point, torch.quint8)
        )

    return layer.get_output(0)


@tensorrt_converter(torch.nn.modules.batchnorm.BatchNorm2d)
def batchnorm2d(network, submod, args, kwargs, layer_name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"BatchNorm2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    return common_batchnorm(network, submod, input_val, layer_name, is_quantized=False)


@tensorrt_converter(torch.nn.quantized.modules.batchnorm.BatchNorm2d)
def quantized_batchnorm2d(network, submod, args, kwargs, layer_name):
    input_val = args[0]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Quantized BatchNorm2d received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    return common_batchnorm(network, submod, input_val, layer_name, is_quantized=True)

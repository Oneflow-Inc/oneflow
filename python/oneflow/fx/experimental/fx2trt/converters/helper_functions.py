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
import tensorrt as trt


def extend_attr_to_tuple(mod: torch.nn.Module, name: str, size: int):
    """
    Extend an attribute of `mod` that named `name` to a tuple of `size`.
    """
    val = getattr(mod, name)
    if not isinstance(val, tuple):
        val = (val,) * size
    return val


def to_numpy(tensor: torch.Tensor):
    """
    Convert a PyTorch Tensor to a Numpy Array.
    """
    if tensor is None:
        return tensor

    if tensor.is_quantized:
        tensor = tensor.dequantize()

    return tensor.cpu().detach().contiguous().numpy()


def get_dyn_range(scale, zero_point, dtype):
    """
    Get the dynamic range of a tensor based on its scale, zero_point and dtype.
    """
    if dtype == torch.quint8:
        min_val, max_val = 0, 255
    elif dtype == torch.qint8:
        min_val, max_val = -128, 127
    else:
        raise RuntimeError(f"Unsupported quantized dtype {dtype}")

    return (min_val - zero_point) * scale, (max_val - zero_point) * scale


def mark_as_int8_layer(layer, dynamic_range):
    """
    Set the precision of a layer to int8 as well as the type of its first output.
    Also set the dynamic range of its first output.
    """
    if layer.type not in {
        trt.LayerType.SHUFFLE,
        trt.LayerType.CONCATENATION,
        trt.LayerType.CONSTANT,
        trt.LayerType.SHAPE,
    }:
        layer.precision = trt.int8

    for i in range(layer.num_outputs):
        output_val = layer.get_output(i)
        output_val.dynamic_range = dynamic_range
        layer.set_output_type(i, trt.int8)
        # output_val.dtype = trt.int8


def get_inputs_from_args_and_kwargs(args, kwargs, input_names):
    inputs = []
    for i, key in enumerate(input_names):
        if key not in kwargs:
            inputs.append(args[i])
        else:
            inputs.append(kwargs[key])
    return inputs

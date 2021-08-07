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
from torch.fx.experimental.fx2trt.fx2trt import tensorrt_converter

from .helper_functions import mark_as_int8_layer


@tensorrt_converter(torch.flatten)
def torch_flatten(network, target, args, kwargs, name):
    # args/kwargs should have already been normalized to kwargs
    assert len(args) == 0
    input_val = kwargs["input"]

    if not isinstance(input_val, trt.tensorrt.ITensor):
        raise RuntimeError(
            f"Flatten received input {input_val} that is not part "
            "of the TensorRT region!"
        )

    # For trt shape we don't have batch dim
    start_dim = kwargs["start_dim"] - 1
    end_dim = len(input_val.shape) if kwargs["end_dim"] == -1 else kwargs["end_dim"] - 1

    assert (
        start_dim >= 0
    ), "Expect non negtive start_dim, this probably due to flatten batch dim."

    new_shape = []
    flatten_dim = 1
    for i, dim in enumerate(input_val.shape):
        if i < start_dim:
            new_shape.append(dim)
        elif i > end_dim:
            new_shape.append(flatten_dim)
            new_shape.append(dim)
        else:
            flatten_dim *= dim

    if end_dim == len(input_val.shape):
        new_shape.append(flatten_dim)

    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(new_shape)
    layer.name = name

    if input_val.dynamic_range:
        mark_as_int8_layer(layer, input_val.dynamic_range)

    return layer.get_output(0)

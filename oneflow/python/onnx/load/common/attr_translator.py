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
from tensorflow.python.framework.tensor_util import MakeNdarray

from oneflow.python.onnx.load.common import data_type

# Keyed by old attribute names.
__tf_attr_translator = {
    "_output_shapes": lambda x: list(
        map(lambda shape: get_tf_shape_as_list(shape.dim), x.list.shape)
    ),
    "shape": lambda x: get_tf_shape_as_list(x.shape.dim),
    "T": lambda x: data_type.tf2onnx(list(x.list.type) or x.type),
    "dtype": lambda x: data_type.tf2onnx(list(x.list.type) or x.type),
    "component_types": lambda x: data_type.tf2onnx(list(x.list.type) or x.type),
    "value": lambda x: MakeNdarray(x.tensor),
    "seed2": lambda x: float(x.i),
    "seed": lambda x: float(x.i),
    "keep_dims": lambda x: int(x.b),
    "squeeze_dims": lambda x: list(x.list.i),
}

__onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: data_type.onnx2tf(x),
    "keepdims": lambda x: bool(x),
    "to": lambda x: data_type.onnx2tf(x),
}


def translate_tf(key, val):
    return __tf_attr_translator.get(key, lambda x: x)(val)


def translate_onnx(key, val):
    return __onnx_attr_translator.get(key, lambda x: x)(val)


def get_tf_shape_as_list(tf_shape_dim):
    return list(map(lambda x: x.size, list(tf_shape_dim)))

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
from typing import Optional, Sequence

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client.nn.module import Module


class TensorBufferToTensor(Module):
    def __init__(self, dtype, instance_shape):
        super().__init__()
        self._op = (
            flow.builtin_op("tensor_buffer_to_tensor")
            .Input("in")
            .Output("out")
            .Attr("dtype", dtype)
            .Attr("instance_shape", instance_shape)
            .Build()
        )

    def forward(self, input):
        return self._op(input)[0]


class TensorToTensorBuffer(Module):
    def __init__(self, instance_dims):
        super().__init__()
        self._op = (
            flow.builtin_op("tensor_to_tensor_buffer")
            .Input("in")
            .Output("out")
            .Attr("instance_dims", instance_dims)
            .Build()
        )

    def forward(self, input):
        return self._op(input)[0]


class GenTensorBuffer(Module):
    def __init__(self, shape, shape_list, value_list, data_type, dynamic_out):
        super().__init__()
        self._op = (
            flow.builtin_op("gen_tensor_buffer")
            .Output("out")
            .Attr("shape", shape)
            .Attr("shape_list", shape_list)
            .Attr("value_list", value_list)
            .Attr("data_type", data_type)
            .Attr("dynamic_out", dynamic_out)
            .Build()
        )

    def forward(self):
        return self._op()[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

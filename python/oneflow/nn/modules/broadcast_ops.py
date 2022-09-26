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
import oneflow as flow
from oneflow.nn.modules.utils import _single, _handle_size_arg


def broadcast_shapes_op(input, *shapes):
    shapes = _handle_size_arg(shapes)
    shapes = _single(shapes)
    return flow._C.broadcast_shapes(input, shapes)


def broadcast_tensors_op(input, *tensors):
    tensors = _handle_size_arg(tensors)
    tensors = _single(tensors)
    return flow._C.broadcast_tensors(input, tensors)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

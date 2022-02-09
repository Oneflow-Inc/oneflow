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
from oneflow.framework.tensor import register_tensor_op


def flip_op(input, dims):
    assert isinstance(dims, (int, list, tuple)), f"dims must be int, list or tuple"
    if isinstance(dims, int):
        dims = [dims]

    input_len = len(input.shape)
    assert len(dims) <= input_len, f"len of dims must less than len of input tensor"
    new_dims = []
    for i in dims:
        if i < 0:
            i += input_len
        assert (
            i < input_len
        ), f"IndexError: Dimension out of range (expected to be in range of {input_len}, but got {i})"
        new_dims.append(i)
    return flow._C.flip(input, new_dims)


if __name__ == "__main__":
    import doctest

    doctest.testmod(raise_on_error=True)

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
import unittest
from collections import OrderedDict
import oneflow.experimental as flow
import numpy as np

from test_util import (
    GenArgList,
    FlattenArray,
    Array2Numpy,
    Index2Coordinate,
)


def gen_numpy_test_sample(input, padding):
    c_idx, h_idx, w_idx = 1, 2, 3
    pad_left = padding[0]
    pad_right = padding[1]
    pad_top = padding[2]
    pad_bottom = padding[3]
    pad_shape = ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right))

    def _np_reflection_pad2d(input, pad_shape):
        numpy_reflect = np.pad(input, pad_shape, "reflect")
        return numpy_reflect

    def _np_reflection_pad2d_grad(src, dest):
        dx_height, dx_width = input.shape[h_idx], input.shape[w_idx]
        dy_height, dy_width = output.shape[h_idx], output.shape[w_idx]

        numpy_src = np.ones(src.shape, np.int32)
        numpy_dest = np.zeros(dest.shape, np.int32)
        array_src = FlattenArray(numpy_src)
        array_dest = FlattenArray(numpy_dest)

        src_num = src.shape[c_idx] * src.shape[h_idx] * src.shape[w_idx]
        dest_num = dest.shape[c_idx] * dest.shape[h_idx] * dest.shape[w_idx]
        elements_num = src.shape[0] * src_num
        for iter_n in range(elements_num):
            coords = Index2Coordinate(iter_n, src.shape)
            n, c, i, j = coords[0], coords[c_idx], coords[h_idx], coords[w_idx]
            ip_x = ip_y = 0
            if j < pad_left:
                ip_x = pad_left * 2 - j
            elif j >= pad_left and j < (dx_width + pad_left):
                ip_x = j
            else:
                ip_x = (dx_width + pad_left - 1) * 2 - j

            if i < pad_top:
                ip_y = pad_top * 2 - i
            elif i >= pad_top and i < (dx_height + pad_top):
                ip_y = i
            else:
                ip_y = (dx_height + pad_top - 1) * 2 - i

            ip_x = ip_x - pad_left
            ip_y = ip_y - pad_top
            src_index = n * src_num + c * dy_width * dy_height + i * dy_width + j
            dest_index = (
                n * dest_num + c * dx_width * dx_height + ip_y * dx_width + ip_x
            )
            array_dest[dest_index] += array_src[src_index]

        numpy_dest = Array2Numpy(array_dest, dest.shape)
        return numpy_dest

    output = _np_reflection_pad2d(input, pad_shape)
    grad = _np_reflection_pad2d_grad(output, input)

    return output, grad


def _test_reflection_pad2d(test_case, shape, padding, device):
    np_input = np.random.randn(*shape).astype(np.float32)

    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )

    if isinstance(padding, int):
        boundary = [padding, padding, padding, padding]

    elif isinstance(padding, tuple) and len(padding) == 4:
        boundary = [padding[0], padding[1], padding[2], padding[3]]
    else:
        raise ValueError("padding must be in or list or tuple!")
    np_out, np_grad = gen_numpy_test_sample(np_input, boundary)

    layer = flow.nn.ReflectionPad2d(padding=padding)
    of_out = layer(of_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))
    of_out = of_out.sum()
    of_out.backward()
    test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestReflectionPad2dModule(flow.unittest.TestCase):
    def test_reflection_pad2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2, 3, 4), (8, 3, 4, 4)]
        arg_dict["padding"] = [(2), (1, 1, 2, 2)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_reflection_pad2d(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

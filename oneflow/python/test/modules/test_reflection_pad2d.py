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

import numpy as np

import oneflow.experimental as flow
from test_util import Array2Numpy, FlattenArray, GenArgList, Index2Coordinate


def gen_numpy_test_sample(input_shape, padding, is_float=True):
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

    if is_float:
        input = np.random.random(input_shape).astype(np.float32)
    else:
        input = np.random.randint(0, 100, input_shape)

    output = _np_reflection_pad2d(input, pad_shape)
    grad = _np_reflection_pad2d_grad(output, input)

    numpy_results = {
        "input": input,
        "padding": padding,
        "output": output,
        "grad": grad,
        "is_float": is_float,
    }

    return numpy_results


def _compare_op_function_with_samples(test_case, device_type, sample):
    layer = flow.nn.ReflectionPad2d(padding=sample["padding"])
    if sample["is_float"]:
        input = flow.Tensor(
            sample["input"],
            dtype=flow.float32,
            device=flow.device(device_type),
            requires_grad=True,
        )
    else:
        input = flow.Tensor(
            sample["input"],
            dtype=flow.int32,
            device=flow.device(device_type),
            requires_grad=True,
        )

    of_out = layer(input)
    test_case.assertTrue(np.allclose(of_out.numpy(), sample["output"], 1e-3, 1e-3))

    of_out = of_out.sum()
    of_out.backward()
    assert np.allclose(input.grad.numpy(), sample["grad"], 1e-3, 1e-3)


def _gen_arg_dict(device_type="cpu"):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = [device_type]
    arg_dict["samples"] = []
    arg_dict["samples"].append(gen_numpy_test_sample((2, 1, 2, 2), [1, 1, 1, 1], True))
    arg_dict["samples"].append(gen_numpy_test_sample((2, 1, 2, 2), [1, 1, 1, 1], False))
    arg_dict["samples"].append(gen_numpy_test_sample((2, 3, 4, 5), [3, 2, 1, 2], True))
    arg_dict["samples"].append(gen_numpy_test_sample((2, 3, 4, 5), [3, 2, 1, 2], False))

    return arg_dict


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestReflectionPad2d1n1d(flow.unittest.TestCase):
    def test_op_function_float_cpu(test_case):
        arg_dict = _gen_arg_dict("cpu")
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)

    def test_op_function_float_gpu(test_case):
        arg_dict = _gen_arg_dict("cuda")
        for arg in GenArgList(arg_dict):
            _compare_op_function_with_samples(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

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
from oneflow.python.ops.array_ops import zeros
from test_util import (
    GenArgList,
    FlattenArray,
    Array2Numpy,
    Index2Coordinate,
)


def _np_replication_pad2d_grad(src, dest, padding):
    c_idx, h_idx, w_idx = 1, 2, 3
    pad_left = padding[0]
    pad_right = padding[1]
    pad_top = padding[2]
    pad_bottom = padding[3]
    dx_height, dx_width = dest.shape[h_idx], dest.shape[w_idx]
    dy_height, dy_width = src.shape[h_idx], src.shape[w_idx]

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
            ip_x = pad_left
        elif j >= pad_left and j < (dx_width + pad_left):
            ip_x = j
        else:
            ip_x = dx_width + pad_left - 1

        if i < pad_top:
            ip_y = pad_top
        elif i >= pad_top and i < (dx_height + pad_top):
            ip_y = i
        else:
            ip_y = dx_height + pad_top - 1

        ip_x = ip_x - pad_left
        ip_y = ip_y - pad_top
        src_index = n * src_num + c * dy_width * dy_height + i * dy_width + j
        dest_index = n * dest_num + c * dx_width * dx_height + ip_y * dx_width + ip_x
        array_dest[dest_index] += array_src[src_index]

    numpy_dest = Array2Numpy(array_dest, dest.shape)
    return numpy_dest


def _test_ReplicationPad2d(test_case, shape, padding, device):
    np_input = np.random.random(shape).astype(np.float32)
    of_input = flow.Tensor(
        np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    if isinstance(padding, int):
        np_boundary = ((0, 0), (0, 0), (padding, padding), (padding, padding))
        boundry = [padding, padding, padding, padding]

    elif isinstance(padding, (tuple, int)) and len(padding) == 4:
        np_boundary = (
            (0, 0),
            (0, 0),
            (padding[2], padding[3]),
            (padding[0], padding[1]),
        )
        boundry = [padding[0], padding[1], padding[2], padding[3]]
    else:
        raise ValueError("padding must be in or list or tuple!")

    layer = flow.nn.ReplicationPad2d(padding=padding)
    of_out = layer(of_input)
    np_out = np.pad(np_input, np_boundary, mode="edge")
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

    of_out = of_out.sum()
    of_out.backward()

    np_out_grad = _np_replication_pad2d_grad(np_out, np_input, boundry)
    test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_out_grad, 1e-3, 1e-3))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestReplicationPad2dModule(flow.unittest.TestCase):
    def test_ReplicationPad2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2, 3, 4), (8, 3, 4, 4)]
        arg_dict["padding"] = [(2), (1, 1, 2, 2)]
        arg_dict["device"] = ["cpu", "cuda"]

        for arg in GenArgList(arg_dict):
            _test_ReplicationPad2d(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

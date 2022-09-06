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
import os
import unittest
import numpy as np


import oneflow as flow
import torch

from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict


def _cmp_expand_stride(
    test_case, input_shape, expand_shape, device="cuda", verbose=False,
):
    input = np.random.randn(*input_shape)
    torch_x = torch.tensor(input, dtype=torch.float32, device=device)
    torch_y = torch_x.expand(*expand_shape)

    x = flow.tensor(input, dtype=flow.float32, device=device)
    y = x.expand(*expand_shape)

    if verbose:
        print("")
        print(f" eager (view::Expand) (device={device}) ".center(50, "="))
        print(f" {input_shape} -> {expand_shape} ".center(50, "*"))
        print(f"x: shape={x.shape}, stride={x.stride()}")
        print(f"y: shape={y.shape}, stride={y.stride()}")
        print(f"torch_y: shape={torch_y.shape}, stride={torch_y.stride()}")
        print(" input ".center(50, "-"))
        print(input)
        print(" y ".center(50, "-"))
        print(y)
        print(" torch_y ".center(50, "-"))
        print(torch_y)

    test_case.assertTrue(np.array_equal(y.stride(), torch_y.stride()))
    test_case.assertTrue(np.array_equal(y.numpy(), torch_y.detach().cpu().numpy()))


def _cmp_expand_non_contiguous_stride(
    test_case, input_shape, perm, expand_shape, device="cuda", verbose=False,
):
    input = np.random.randn(*input_shape).astype(np.float32)
    x = flow.tensor(input, device=device)
    y = x.permute(*perm)
    z = y.expand(*expand_shape)

    torch_x = torch.tensor(input, device=device)
    torch_y = torch_x.permute(*perm)
    torch_z = torch_y.expand(*expand_shape)

    if verbose:
        print("")
        print(f" non_contiguous (device={device}) ".center(50, "-"))
        print(f" {input_shape}, {perm} -> {expand_shape} ".center(50, "-"))
        print(f"x: shape={x.shape}, stride={x.stride()}")
        print(f"y: shape={y.shape}, stride={y.stride()}")
        print(f"z: shape={z.shape}, stride={z.stride()}")
        print(f"torch_y: shape={torch_y.shape}, stride={torch_y.stride()}")
        print(f"torch_z: shape={torch_z.shape}, stride={torch_z.stride()}")
        print(" input ".center(50, "-"))
        print(input)
        print(" z ".center(50, "-"))
        print(z)
        print(" torch_z ".center(50, "-"))
        print(torch_z)

    test_case.assertTrue(np.array_equal(z.stride(), torch_z.stride()))
    test_case.assertTrue(np.array_equal(z.numpy(), torch_z.detach().cpu().numpy()))


def _cmp_lazy_expand_stride(
    test_case, input_shape, expand_shape, device="cuda", verbose=False,
):
    input = np.random.randn(*input_shape)
    torch_x = torch.tensor(input, dtype=torch.float32, device=device)
    torch_y = torch_x.expand(*expand_shape).contiguous()
    # oneflow lazy must do this contiguous

    class MyGraph(flow.nn.Graph):
        def __init__(self, expand_shape):
            super().__init__()
            self.expand_shape = expand_shape

        def build(self, x):
            return x.expand(*self.expand_shape)

    expand_graph = MyGraph(expand_shape)
    x = flow.tensor(input, dtype=flow.float32, device=device)
    y = expand_graph(x)

    squeeze_y_stride = []
    for d, s in zip(y.shape, y.stride()):
        if d != 1:
            squeeze_y_stride.append(s)

    squeeze_torch_y_stride = []
    for d, s in zip(torch_y.shape, torch_y.stride()):
        if d != 1:
            squeeze_torch_y_stride.append(s)

    if verbose:
        print("")
        print(f" lazy (expand op/kernel) (device={device}) ".center(50, "="))
        print(f" {input_shape} -> {expand_shape} ".center(50, "*"))
        print(f"x: shape={x.shape}, stride={x.stride()}")
        print(f"y: shape={y.shape}, stride={y.stride()}")
        print(f"torch_y: shape={torch_y.shape}, stride={torch_y.stride()}")
        print(f"squeeze_y_stride={squeeze_y_stride}")
        print(f"squeeze_torch_y_stride={squeeze_torch_y_stride}")
        print(" input ".center(50, "-"))
        print(input)
        print(" y ".center(50, "-"))
        print(y)
        print(" torch_y ".center(50, "-"))
        print(torch_y)

    test_case.assertTrue(np.array_equal(squeeze_y_stride, squeeze_torch_y_stride))
    test_case.assertTrue(np.array_equal(y.numpy(), torch_y.detach().cpu().numpy()))


@flow.unittest.skip_unless_1n1d()
class ExpandStrideTestCase(flow.unittest.TestCase):
    test_shape_tuple_list = [
        ((1, 2), (2, 2)),
        ((1, 2), (1, 1, 2)),
        ((1, 2), (1, 2, 2)),
        ((1, 2), (2, 1, 2)),
        ((1, 2), (2, 2, 2)),
        ((1, 2), (1, 1, 1, 2)),
        ((1, 2), (1, 2, 1, 2)),
        ((1, 2), (2, 1, 1, 2)),
        ((1, 2), (2, 2, 1, 2)),
        ((1, 2), (2, 2, 2, 2)),
        ((2, 1), (2, 2)),
        ((2, 1), (1, 2, 1)),
        ((2, 1), (1, 2, 2)),
        ((2, 1), (2, 2, 1)),
        ((2, 1), (2, 2, 2)),
        ((2, 1), (1, 1, 2, 1)),
        ((2, 1), (1, 2, 2, 1)),
        ((2, 1), (2, 2, 2, 1)),
        ((2, 1), (2, 2, 2, 2)),
        ((2, 2), (1, 2, 2)),
        ((2, 2), (2, 2, 2)),
        ((2, 2), (1, 1, 2, 2)),
        ((2, 2), (1, 2, 2, 2)),
        ((2, 2), (2, 1, 2, 2)),
        ((2, 2), (2, 2, 2, 2)),
        ((2, 1, 4), (2, 2, 2, 4)),
        ((2, 1, 3), (2, 1, -1, -1, -1)),
        ((2, 1, 3), (1, 2, -1, -1, -1)),
        ((2, 1, 3), (2, 2, -1, -1, -1)),
        ((2, 1, 3), (2, 1, -1, 2, 3)),
        ((2, 1, 3), (1, 2, 2, 2, -1)),
        ((2, 1, 3), (2, 2, 2, 2, 3)),
        ((2, 3, 4), (1, 2, -1, -1, -1)),
        ((2, 3, 4), (2, 1, -1, -1, -1)),
        ((2, 3, 4), (2, 2, -1, -1, -1)),
        ((), (1,)),
        ((), (2,)),
        ((), (1, 2)),
        ((), (2, 1)),
        ((), (2, 2)),
    ]

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_on_cpu(self):
        arg_dict = OrderedDict()
        arg_dict["verbose"] = [False]
        arg_dict["device"] = ["cpu"]
        arg_dict["shapes"] = self.test_shape_tuple_list
        for kwargs in GenArgDict(arg_dict):
            assert "shapes" in kwargs
            input_shape, expand_shape = kwargs.pop("shapes")
            _cmp_expand_stride(self, input_shape, expand_shape, **kwargs)

    def test_stride(self):
        arg_dict = OrderedDict()
        arg_dict["verbose"] = [False]
        arg_dict["device"] = ["cuda"]
        arg_dict["shapes"] = self.test_shape_tuple_list
        for kwargs in GenArgDict(arg_dict):
            assert "shapes" in kwargs
            input_shape, expand_shape = kwargs.pop("shapes")
            _cmp_expand_stride(self, input_shape, expand_shape, **kwargs)

    def test_non_contiguous_stride(self):
        arg_dict = OrderedDict()
        arg_dict["verbose"] = [False]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shapes"] = [
            ((2, 1, 3), (0, 2, 1), (1, 2, -1, -1, -1)),
            ((2, 1, 3), (0, 2, 1), (2, 1, -1, -1, -1)),
            ((2, 1, 3), (0, 2, 1), (2, 3, -1, -1, -1)),
            ((2, 1, 3), (0, 2, 1), (1, 2, -1, -1, 2)),
            ((2, 1, 3), (0, 2, 1), (2, 1, -1, -1, 2)),
            ((2, 1, 3), (0, 2, 1), (2, 3, -1, -1, 2)),
            ((2, 3, 4), (0, 2, 1), (1, 2, -1, -1, -1)),
            ((2, 3, 4), (0, 2, 1), (2, 1, -1, -1, -1)),
            ((2, 3, 4), (0, 2, 1), (2, 2, -1, -1, -1)),
        ]
        for kwargs in GenArgDict(arg_dict):
            assert "shapes" in kwargs
            input_shape, perm, expand_shape = kwargs.pop("shapes")
            _cmp_expand_non_contiguous_stride(
                self, input_shape, perm, expand_shape, **kwargs
            )

    def test_lazy(self):
        arg_dict = OrderedDict()
        arg_dict["verbose"] = [False]
        arg_dict["device"] = ["cuda"]
        arg_dict["shapes"] = self.test_shape_tuple_list
        for kwargs in GenArgDict(arg_dict):
            assert "shapes" in kwargs
            input_shape, expand_shape = kwargs.pop("shapes")
            _cmp_lazy_expand_stride(self, input_shape, expand_shape, **kwargs)


if __name__ == "__main__":
    unittest.main()

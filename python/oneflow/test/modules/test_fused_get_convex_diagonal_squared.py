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
import torch
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def torch_fused_get_convex_diagonal_squared(
    b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2, eps
):
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    return c2


def _test_fused_get_convex_diagonal_squared_impl(test_case, device, shape):
    def compare(a, b, rtol=1e-5, atol=1e-5):
        test_case.assertTrue(
            np.allclose(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=rtol, atol=atol
            ),
            f"\na\n{a.detach().cpu().numpy()}\n{'-' * 80}\nb:\n{b.detach().cpu().numpy()}\n{'*' * 80}\ndiff:\n{a.detach().cpu().numpy() - b.detach().cpu().numpy()}",
        )

    eps = 1e-8
    x = []
    torch_x = []
    for _ in range(8):
        tmp = flow.tensor(
            np.random.randn(*shape),
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=True,
        )
        x.append(tmp)
        torch_x.append(
            torch.tensor(
                tmp.numpy(),
                dtype=torch.float32,
                device=torch.device(device),
                requires_grad=True,
            )
        )
    b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2 = (
        x[0],
        x[1],
        x[2],
        x[3],
        x[4],
        x[5],
        x[6],
        x[7],
    )
    (
        torch_b1_x1,
        torch_b1_x2,
        torch_b2_x1,
        torch_b2_x2,
        torch_b1_y1,
        torch_b1_y2,
        torch_b2_y1,
        torch_b2_y2,
    ) = (
        torch_x[0],
        torch_x[1],
        torch_x[2],
        torch_x[3],
        torch_x[4],
        torch_x[5],
        torch_x[6],
        torch_x[7],
    )
    c2 = flow._C.fused_get_convex_diagonal_squared(
        b1_x1, b1_x2, b2_x1, b2_x2, b1_y1, b1_y2, b2_y1, b2_y2, eps
    )
    torch_c2 = torch_fused_get_convex_diagonal_squared(
        torch_b1_x1,
        torch_b1_x2,
        torch_b2_x1,
        torch_b2_x2,
        torch_b1_y1,
        torch_b1_y2,
        torch_b2_y1,
        torch_b2_y2,
        eps,
    )
    compare(c2, torch_c2)

    c2.sum().backward()
    torch_c2.sum().backward()
    compare(b1_x1.grad, torch_b1_x1.grad)
    compare(b1_x2.grad, torch_b1_x2.grad)
    compare(b2_x1.grad, torch_b2_x1.grad)
    compare(b2_x2.grad, torch_b2_x2.grad)
    compare(b1_y1.grad, torch_b1_y1.grad)
    compare(b1_y2.grad, torch_b1_y2.grad)
    compare(b2_y1.grad, torch_b2_y1.grad)
    compare(b2_y2.grad, torch_b2_y2.grad)


@flow.unittest.skip_unless_1n1d()
class TestGetCenterDistModule(flow.unittest.TestCase):
    def test_fused_get_convex_diagonal_squared(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_fused_get_convex_diagonal_squared_impl]
        arg_dict["device"] = ["cuda"]
        arg_dict["shape"] = [(583, 1), (759, 1), (1234, 1)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

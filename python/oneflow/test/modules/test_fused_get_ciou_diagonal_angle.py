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

import math
import numpy as np
import torch
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def torch_get_ciou_diagonal_angle(w1, h1, w2, h2, eps=1e-8):
    return (4 / math.pi ** 2) * torch.pow(
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
    )


def _test_fused_get_ciou_diagonal_angle_impl(test_case, device, shape):
    def compare(a, b, rtol=1e-5, atol=1e-5):
        test_case.assertTrue(
            np.allclose(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=rtol, atol=atol
            ),
            f"\na\n{a.detach().cpu().numpy()}\n{'-' * 80}\nb:\n{b.detach().cpu().numpy()}\n{'*' * 80}\ndiff:\n{a.detach().cpu().numpy() - b.detach().cpu().numpy()}",
        )

    x = []
    torch_x = []
    for _ in range(4):
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
    w1, h1, w2, h2 = (
        x[0],
        x[1],
        x[2],
        x[3],
    )
    (torch_w1, torch_h1, torch_w2, torch_h2,) = (
        torch_x[0],
        torch_x[1],
        torch_x[2],
        torch_x[3],
    )
    v = flow._C.fused_get_ciou_diagonal_angle(w1, h1, w2, h2, eps=1e-8)
    torch_v = torch_get_ciou_diagonal_angle(torch_w1, torch_h1, torch_w2, torch_h2,)
    compare(v, torch_v)

    v.sum().backward()
    torch_v.sum().backward()
    compare(w1.grad, torch_w1.grad)
    compare(h1.grad, torch_h1.grad)
    compare(w2.grad, torch_w2.grad)
    compare(h2.grad, torch_h2.grad)


@flow.unittest.skip_unless_1n1d()
class TestGetCiouDiagonalAngle(flow.unittest.TestCase):
    def test_fused_get_ciou_diagonal_angle(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_fused_get_ciou_diagonal_angle_impl]
        arg_dict["device"] = ["cuda"]
        arg_dict["shape"] = [(583, 1), (759, 1), (1234, 1)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

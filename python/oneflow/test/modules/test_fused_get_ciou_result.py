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


def _test_get_ciou_result_impl(test_case, device, shape):
    eps = 1e-7
    x = []
    torch_x = []
    for _ in range(4):
        tmp = flow.tensor(
            np.random.uniform(0, 1, shape),
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
    v, iou, rho2, c2 = x[0], x[1], x[2], x[3]
    y = flow._C.fused_get_ciou_result(v, iou, rho2, c2, eps)[0]
    torch_v, torch_iou, torch_rho2, torch_c2 = (
        torch_x[0],
        torch_x[1],
        torch_x[2],
        torch_x[3],
    )
    with torch.no_grad():
        torch_alpha = torch_v / (torch_v - torch_iou + (1.0 + eps))
    torch_y = torch_iou - (torch_rho2 / torch_c2 + torch_v * torch_alpha)

    def compare(a, b, rtol=1e-5, atol=1e-5):
        test_case.assertTrue(
            np.allclose(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=rtol, atol=atol
            ),
            f"\na\n{a.detach().cpu().numpy()}\n{'-' * 80}\nb:\n{b.detach().cpu().numpy()}\n{'*' * 80}\ndiff:\n{a.detach().cpu().numpy() - b.detach().cpu().numpy()}",
        )

    compare(y, torch_y)

    res = y.sum()
    torch_res = torch_y.sum()
    res.backward()
    torch_res.backward()
    compare(v.grad, torch_v.grad)
    compare(iou.grad, torch_iou.grad)
    compare(rho2.grad, torch_rho2.grad)
    compare(c2.grad, torch_c2.grad)


@flow.unittest.skip_unless_1n1d()
class TestGetCiouResultModule(flow.unittest.TestCase):
    def test_get_ciou_result(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_get_ciou_result_impl]
        arg_dict["device"] = ["cuda"]
        arg_dict["shape"] = [(492), (691, 1), (1162, 1)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

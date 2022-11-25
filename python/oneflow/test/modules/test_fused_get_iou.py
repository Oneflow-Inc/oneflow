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


def _test_get_iou_impl(test_case, device, shape):
    eps = 1e-7
    x = []
    torch_x = []
    for _ in range(5):
        tmp = flow.tensor(
            np.random.uniform(0, 1, shape),
            dtype=flow.float64,
            device=flow.device(device),
            requires_grad=True if (_ < 2 or _ > 3) else False,
        )
        x.append(tmp)
        torch_x.append(
            torch.tensor(
                tmp.numpy(),
                dtype=torch.float64,
                device=torch.device(device),
                requires_grad=True if (_ < 2 or _ > 3) else False,
            )
        )
    w1, h1, w2, h2, inter = x[0], x[1], x[2], x[3], x[4]
    iou = flow._C.fused_get_iou(w1, h1, w2, h2, inter, eps)
    torch_w1, torch_h1, torch_w2, torch_h2, torch_inter = (
        torch_x[0],
        torch_x[1],
        torch_x[2],
        torch_x[3],
        torch_x[4],
    )
    torch_iou = torch_inter / (
        torch_w1 * torch_h1 + torch_w2 * torch_h2 - torch_inter + eps
    )

    def compare(a, b, rtol=1e-5, atol=1e-5, w1=w1, h1=h1, w2=w2, h2=h2, inter=inter):
        test_case.assertTrue(
            np.allclose(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=rtol, atol=atol
            ),
            f"\na\n{a.detach().cpu().numpy()}\n{'-' * 80}\nb:\n{b.detach().cpu().numpy()}\n{'*' * 80}\ndiff:\n{a.detach().cpu().numpy() - b.detach().cpu().numpy()}",
        )

    compare(iou, torch_iou)

    res = iou.sum()
    torch_res = torch_iou.sum()
    res.backward()
    torch_res.backward()
    compare(w1.grad, torch_w1.grad)
    compare(h1.grad, torch_h1.grad)
    compare(inter.grad, torch_inter.grad)


@flow.unittest.skip_unless_1n1d()
class TestGetIouModule(flow.unittest.TestCase):
    def test_get_iou(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_get_iou_impl]
        arg_dict["device"] = ["cuda"]
        arg_dict["shape"] = [(492), (691, 1), (1162, 1)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

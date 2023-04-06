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


def _test_get_boundding_boxes_coord_impl(test_case, device, shape):
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
    x1, y1, w1, h1, x2, y2, w2, h2 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
    (
        b1_x1,
        b1_x2,
        b1_y1,
        b1_y2,
        b2_x1,
        b2_x2,
        b2_y1,
        b2_y2,
    ) = flow._C.fused_get_boundding_boxes_coord(x1, y1, w1, h1, x2, y2, w2, h2)
    torch_x1, torch_y1, torch_w1, torch_h1, torch_x2, torch_y2, torch_w2, torch_h2 = (
        torch_x[0],
        torch_x[1],
        torch_x[2],
        torch_x[3],
        torch_x[4],
        torch_x[5],
        torch_x[6],
        torch_x[7],
    )
    torch_w1_, torch_h1_, torch_w2_, torch_h2_ = (
        torch_w1 / 2,
        torch_h1 / 2,
        torch_w2 / 2,
        torch_h2 / 2,
    )
    torch_b1_x1, torch_b1_x2, torch_b1_y1, torch_b1_y2 = (
        torch_x1 - torch_w1_,
        torch_x1 + torch_w1_,
        torch_y1 - torch_h1_,
        torch_y1 + torch_h1_,
    )
    torch_b2_x1, torch_b2_x2, torch_b2_y1, torch_b2_y2 = (
        torch_x2 - torch_w2_,
        torch_x2 + torch_w2_,
        torch_y2 - torch_h2_,
        torch_y2 + torch_h2_,
    )

    def compare(a, b, rtol=1e-5, atol=1e-8):
        test_case.assertTrue(
            np.allclose(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=rtol, atol=atol
            ),
            f"\na\n{a.detach().cpu().numpy()}\n{'-' * 80}\nb:\n{b.detach().cpu().numpy()}\n{'*' * 80}\ndiff:\n{a.detach().cpu().numpy() - b.detach().cpu().numpy()}",
        )

    compare(b1_x1, torch_b1_x1)
    compare(b1_x2, torch_b1_x2)
    compare(b1_y1, torch_b1_y1)
    compare(b1_y2, torch_b1_y2)
    compare(b2_x1, torch_b2_x1)
    compare(b2_x2, torch_b2_x2)
    compare(b2_y1, torch_b2_y1)
    compare(b2_y2, torch_b2_y2)
    res = (
        (b1_x1 + 2 * b1_x2 + b1_y1 + b1_y2 + b2_x1 + b2_x2 + b2_y1 + b2_y2) * 2
    ).sum()
    torch_res = (
        (
            torch_b1_x1
            + 2 * torch_b1_x2
            + torch_b1_y1
            + torch_b1_y2
            + torch_b2_x1
            + torch_b2_x2
            + torch_b2_y1
            + torch_b2_y2
        )
        * 2
    ).sum()
    res.sum().backward()
    torch_res.sum().backward()
    compare(x1.grad, torch_x1.grad)
    compare(y1.grad, torch_y1.grad)
    compare(w1.grad, torch_w1.grad)
    compare(h1.grad, torch_h1.grad)
    compare(x2.grad, torch_x2.grad)
    compare(y2.grad, torch_y2.grad)
    compare(w2.grad, torch_w2.grad)
    compare(h2.grad, torch_h2.grad)


@flow.unittest.skip_unless_1n1d()
class TestGetBounddingBoxesCoordModule(flow.unittest.TestCase):
    def test_get_boundding_boxes_coord(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_get_boundding_boxes_coord_impl]
        arg_dict["device"] = ["cuda"]
        arg_dict["shape"] = [(583, 1), (759, 1), (1234, 1)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

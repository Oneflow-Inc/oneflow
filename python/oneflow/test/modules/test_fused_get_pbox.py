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


def torch_get_pbox(pxy, pwh, anchors_i):
    pxy = pxy.sigmoid() * 2 - 0.5
    pwh = (pwh.sigmoid() * 2) ** 2 * anchors_i
    return torch.cat([pxy, pwh], dim=1)


def _test_fused_get_pbox_impl(test_case, device, shape):
    def compare(a, b, rtol=1e-5, atol=1e-5):
        test_case.assertTrue(
            np.allclose(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=rtol, atol=atol
            ),
            f"\na\n{a.detach().cpu().numpy()}\n{'-' * 80}\nb:\n{b.detach().cpu().numpy()}\n{'*' * 80}\ndiff:\n{a.detach().cpu().numpy() - b.detach().cpu().numpy()}\n",
        )

    x = []
    torch_x = []
    for _ in range(3):
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
    pxy, pwh, anchors_i = x[0], x[1], x[2]
    torch_pxy, torch_pwh, torch_anchors_i = torch_x[0], torch_x[1], torch_x[2]
    pbox = flow._C.fused_get_pbox(pxy, pwh, anchors_i)
    torch_pbox = torch_get_pbox(torch_pxy, torch_pwh, torch_anchors_i)
    compare(pbox, torch_pbox)

    pbox.sum().backward()
    torch_pbox.sum().backward()
    compare(pxy.grad, torch_pxy.grad)
    compare(pwh.grad, torch_pwh.grad)
    compare(anchors_i.grad, torch_anchors_i.grad)


@flow.unittest.skip_unless_1n1d()
class TestGetPboxModule(flow.unittest.TestCase):
    def test_fused_get_pbox(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_fused_get_pbox_impl]
        arg_dict["device"] = ["cuda"]
        arg_dict["shape"] = [(4, 1234), (24, 583), (128, 73)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

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


def _test_multi_tensor_weight_update_impl(test_case, device, shape, n, d):
    def compare(a, b, rtol=1e-5, atol=1e-5):
        test_case.assertTrue(
            np.allclose(
                a.detach().cpu().numpy(), b.detach().cpu().numpy(), rtol=rtol, atol=atol
            ),
            f"\na\n{a.detach().cpu().numpy()}\n{'-' * 80}\nb:\n{b.detach().cpu().numpy()}\n{'*' * 80}\ndiff:\n{a.detach().cpu().numpy() - b.detach().cpu().numpy()}",
        )

    weight = []
    torch_weight = []
    weight_update = []
    torch_weight_update = []
    for _ in range(n):
        tmp = flow.tensor(
            np.random.randn(*shape),
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=False,
        )
        weight.append(tmp)
        torch_weight.append(
            torch.tensor(
                tmp.numpy(),
                dtype=torch.float32,
                device=torch.device(device),
                requires_grad=False,
            )
        )
        tmp = flow.tensor(
            np.random.randn(*shape),
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=False,
        )
        weight_update.append(tmp)
        torch_weight_update.append(
            torch.tensor(
                tmp.numpy(),
                dtype=torch.float32,
                device=torch.device(device),
                requires_grad=False,
            )
        )
    for i, v in enumerate(torch_weight):
        v = v * d
        v = v + (1 - d) * torch_weight_update[i]
        torch_weight[i] = v

    flow._C.multi_tensor_yolov5_weight_update(weight, weight_update, d)
    for i in range(n):
        compare(weight[i], torch_weight[i])


@flow.unittest.skip_unless_1n1d()
class TestMultiTensorWeightUpdateModule(flow.unittest.TestCase):
    def test_multi_tensor_weight_update(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_multi_tensor_weight_update_impl]
        arg_dict["device"] = ["cuda"]
        arg_dict["shape"] = [(20, 1), (30, 1), (55, 1)]
        arg_dict["n"] = [5, 10, 292]
        arg_dict["d"] = [0.22, 0.5]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

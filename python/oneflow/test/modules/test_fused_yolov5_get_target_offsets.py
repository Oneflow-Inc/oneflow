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


def torch_get_target_offsets(gxy, gxi, g):
    j, k = ((gxy % 1 < g) & (gxy > 1)).T
    l, m = ((gxi % 1 < g) & (gxi > 1)).T
    return torch.stack((j, k, l, m, torch.ones_like(j)))


def _test_fused_get_target_offsets_impl(test_case, device, shape, g):
    x = []
    torch_x = []
    for _ in range(3):
        tmp = flow.tensor(
            np.random.randn(*shape),
            dtype=flow.float32,
            device=flow.device(device),
            requires_grad=False,
        )
        x.append(tmp)
        torch_x.append(
            torch.tensor(
                tmp.numpy(),
                dtype=torch.float32,
                device=torch.device(device),
                requires_grad=False,
            )
        )
    gxy, gxi = x[0] + x[1], x[0] + x[2]
    torch_gxy, torch_gxi = torch_x[0] + torch_x[1], torch_x[0] + torch_x[2]
    j = flow._C.fused_yolov5_get_target_offsets(gxy, gxi, g)
    torch_j = torch_get_target_offsets(torch_gxy, torch_gxi, g)
    test_case.assertTrue((j.cpu().numpy() == torch_j.cpu().numpy()).any())


@flow.unittest.skip_unless_1n1d()
class TestGetTargetOffsetsModule(flow.unittest.TestCase):
    def test_fused_get_target_offsets_area(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_fused_get_target_offsets_impl]
        arg_dict["device"] = ["cuda"]
        arg_dict["shape"] = [(1234, 2), (583, 2), (128, 2)]
        arg_dict["g"] = [0.1, 0.5, 0.9]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

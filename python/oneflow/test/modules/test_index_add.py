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
import numpy as np
import torch as torch_origin
from collections import OrderedDict

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList
import unittest
from oneflow.test_utils.automated_test_util import *


def _test_index_add(test_case, device):
    torch_origin_x = torch_origin.ones(5, 3).to(device)
    torch_origin_t = torch_origin.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch_origin.float
    ).to(device)
    torch_origin_index = torch_origin.tensor([0, 4, 2]).to(device)
    torch_origin_y = torch_origin.index_add(
        torch_origin_x, 0, torch_origin_index, torch_origin_t
    )
    torch_origin_y_alpha = torch_origin.index_add(
        torch_origin_x, 0, torch_origin_index, torch_origin_t, alpha=-1
    )

    flow_x = flow.ones(5, 3).to(device)
    flow_t = flow.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=flow.float).to(device)
    flow_index = flow.tensor([0, 4, 2]).to(device)
    flow_y = flow.index_add(flow_x, 0, flow_index, flow_t)
    flow_y_alpha = flow.index_add(flow_x, 0, flow_index, flow_t, alpha=-1)
    test_case.assertTrue(
        np.allclose(torch_origin_y.cpu().numpy(), flow_y.cpu().numpy(), 1e-05, 1e-05)
    )
    test_case.assertTrue(
        np.allclose(
            torch_origin_y_alpha.cpu().numpy(), flow_y_alpha.cpu().numpy(), 1e-05, 1e-05
        )
    )

    # check inplace
    torch_origin_x.index_add_(0, torch_origin_index, torch_origin_t)
    flow_x.index_add_(0, flow_index, flow_t)
    test_case.assertTrue(
        np.allclose(torch_origin_y.cpu().numpy(), flow_y.cpu().numpy(), 1e-05, 1e-05)
    )


@flow.unittest.skip_unless_1n1d()
class TestIndexAdd(flow.unittest.TestCase):
    def test_index_add(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_index_add]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @profile(torch.index_add)
    def profile_index_add(test_case):
        torch.index_add(
            torch.ones(50, 30),
            0,
            torch.arange(30),
            torch.arange(1, 901, dtype=torch.float32).reshape(30, 30),
        )


if __name__ == "__main__":
    unittest.main()

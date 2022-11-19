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
import torch
from collections import OrderedDict

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList
import unittest


def _test_index_add(test_case, device):
    torch_x = torch.ones(5, 3).to(device)
    torch_t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float).to(
        device
    )
    torch_index = torch.tensor([0, 4, 2]).to(device)
    torch_y = torch.index_add(torch_x, 0, torch_index, torch_t)

    flow_x = flow.ones(5, 3).to(device)
    flow_t = flow.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=flow.float).to(device)
    flow_index = flow.tensor([0, 4, 2]).to(device)
    flow_y = flow.index_add(flow_x, 0, flow_index, flow_t)
    print(flow_y)


@flow.unittest.skip_unless_1n1d()
class TestIndexAdd(flow.unittest.TestCase):
    def test_index_add(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_index_add]
        arg_dict["device"] = ["cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

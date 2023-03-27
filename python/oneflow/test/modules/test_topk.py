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
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import torch
import oneflow.unittest


def _test_top_k(test_case, shape, k, dim, device):
    if k >= shape[dim]:
        return
    x_np = np.random.randn(*shape)
    x_of = flow.tensor(x_np, device=device)
    of_out = flow.topk(x_of, k=k, dim=dim)
    x_pt = torch.tensor(x_np, device=device)
    pt_out = torch.topk(x_pt, k=k, dim=dim)
    test_case.assertTrue(
        np.array_equal(of_out.values.cpu().numpy(), pt_out.values.cpu().numpy())
    )
    test_case.assertTrue(
        np.array_equal(of_out.indices.cpu().numpy(), pt_out.indices.cpu().numpy())
    )


@flow.unittest.skip_unless_1n1d()
class TestTopK(flow.unittest.TestCase):
    def test_in_top_k(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 16), (1, 1024), (8, 8), (8, 256)]
        arg_dict["k"] = [1, 4, 64]
        arg_dict["dim"] = [0, 1]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_top_k(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

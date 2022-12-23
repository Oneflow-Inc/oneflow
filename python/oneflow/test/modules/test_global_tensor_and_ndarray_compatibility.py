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


import torch
import oneflow as flow
import unittest
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


import unittest
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *
import numpy as np

test_compute_op_list = [
    "+",
    # "-",
    # "*",
    # "/",
    # "**",
    # "//",
    # "%",
]


def do_test_compute_op(test_case, ndim, placement, sbp):
    dims = [random(1, 2) * 2 for i in range(ndim)]
    x = random_tensor(ndim, *dims,dtype=int,low=0,high=5)
    x = x.to_global(placement=placement, sbp=sbp)
    flow_input = x.oneflow.detach()
    torch_input = x.pytorch.detach()
    random_numpy =  np.random.randint(1,5)

    for op in test_compute_op_list:
        z_flow = eval(f"flow_input {op} random_numpy")
        z_torch = eval(f"torch_input {op} random_numpy")
        print("z_flow:",z_flow)
        print("z_torch:",z_torch)
        print("flow_input:",flow_input)
        print("torch_input:",torch_input)
        print(random_numpy)
        print(op)
        print("\n")
        test_case.assertTrue(np.allclose(z_flow.numpy(), z_torch.numpy()))


class TestGlobal(flow.unittest.TestCase):
    @globaltest
    def test_div(test_case):
        # random ndim in range [1,4]
        ndim = random(1, 5).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                do_test_compute_op(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()

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

# x = flow.tensor([10, 10, 10])
# x_torch = torch.tensor([10, 10, 10])
# import numpy as np

# a = np.array([1, 2, 10])

# #z = x == a #
# z_torch = a*x_torch  #

# #print(z)
# print(z_torch)

# print("all success")

import unittest
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *
import numpy as np

test_compute_op_list = [
    "+",
    "-",
    "*",
    "/",
    "**",
    "//",
    "%",
]



def do_test_compute_op(test_case, ndim, placement, sbp):
    dims = [random(1, 4) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims)
    x = x.to_global(placement=placement, sbp=sbp)
    flow_input = x.oneflow.detach()
    torch_input = x.pytorch.detach()
    random_numpy = np.random.randn()
    for op in test_compute_op_list:
        z_flow = eval(f"torch_input {op} random_numpy")
        z_torch = eval(f"torch_input {op} random_numpy")
        test_case.assertEqual(z_flow.numpy().all(), z_torch.numpy().all())



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

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
import numpy as np
from collections import OrderedDict

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList


def _test_gather_nd(test_case, ndim, indices, placement, sbp):
    dims = [8 for _ in range(ndim)]
    input = random_tensor(ndim, *dims).oneflow
    indices = torch.tensor(indices).oneflow

    global_input = (
        input.clone().to_global(placement=placement, sbp=sbp).requires_grad_()
    )
    global_input.retain_grad()
    global_indices = indices.to_global(
        placement=placement, sbp=[flow.sbp.broadcast for _ in range(len(sbp))]
    )
    global_output = flow.gather_nd(global_input, global_indices)

    device = placement.type
    local_input = input.to_local().clone().to(device)
    local_input.retain_grad()
    local_indices = indices.to(device)
    local_output = flow.gather_nd(local_input, local_indices)

    global_output.sum().backward()
    local_output.sum().backward()

    test_case.assertTrue(np.array_equal(global_output.numpy(), local_output.numpy()))
    test_case.assertTrue(
        np.array_equal(global_input.grad.numpy(), local_input.grad.numpy())
    )


class TestGather_nd(flow.unittest.TestCase):
    @globaltest
    def test_gather_nd(test_case):
        arg_dict = OrderedDict()
        arg_dict["ndim"] = [2, 3]
        arg_dict["indices"] = [[[0], [2]], [[0, 2], [2, 1]]]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=2):
                    _test_gather_nd(test_case, *arg, placement, sbp)


if __name__ == "__main__":
    unittest.main()

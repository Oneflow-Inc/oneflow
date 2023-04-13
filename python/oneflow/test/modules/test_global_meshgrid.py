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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(n=5, auto_backward=False, check_graph=True)
def _test_meshgrid(test_case, placement):
    x_sbp = random_sbp(placement, max_dim=1)
    x = random_tensor(ndim=1, dim0=8, requires_grad=False).to_global(placement, x_sbp)
    y_sbp = random_sbp(placement, max_dim=1)
    y = random_tensor(ndim=1, dim0=8, requires_grad=False).to_global(placement, y_sbp)
    res = torch.meshgrid(x, y)
    return res[0], res[1]


class TestMeshGrid(flow.unittest.TestCase):
    @globaltest
    def test_meshgrid(test_case):
        for placement in all_placement():
            _test_meshgrid(test_case, placement)


if __name__ == "__main__":
    unittest.main()

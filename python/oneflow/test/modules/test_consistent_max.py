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


@autotest(n=1, check_graph=False)
def _test_max(test_case, placement, sbp):
    keepdim = random().to(bool)
    dim = random(0, 4).to(int)
    dim0 = random().to(int).value() * 8
    dim1 = random().to(int).value() * 8
    x = random_pytorch_tensor(ndim=4, dim0=dim0, dim1=dim1).to_consistent(
        placement, sbp
    )
    y = torch.max(x, dim=dim, keepdim=keepdim)
    max_value = y[0]
    return max_value


class TestMaxModule(flow.unittest.TestCase):
    @consistent
    def test_max(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_max(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

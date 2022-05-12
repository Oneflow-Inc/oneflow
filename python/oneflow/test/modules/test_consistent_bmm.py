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
def _test_bmm_with_random_data(test_case, placement, sbp):
    batch = random(1, 3).to(int) * 8
    m = random(1, 3).to(int) * 8
    n = random(1, 3).to(int) * 8
    k = random(1, 3).to(int) * 8
    x = random_tensor(ndim=3, dim0=batch, dim1=m, dim2=k).to_global(
        placement=placement, sbp=sbp
    )
    y = random_tensor(ndim=3, dim0=batch, dim1=k, dim2=n).to_global(
        placement=placement, sbp=sbp
    )
    return torch.bmm(x, y)


class TestModule(flow.unittest.TestCase):
    @globaltest
    def test_bmm_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_bmm_with_random_data(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

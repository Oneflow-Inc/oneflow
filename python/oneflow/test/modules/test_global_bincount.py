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


@autotest(n=1, check_graph=False, auto_backward=False)
def _test_bincount(test_case, placement, sbp):
    x = random_tensor(1, 64, low=0, dtype=int).to_global(placement=placement, sbp=sbp)
    weight = random_tensor(1, 64).to_global(placement=placement, sbp=sbp)
    minlength = random(1, 100).to(int)
    return torch.bincount(x, weight, minlength)


class TestBinCountModule(flow.unittest.TestCase):
    @globaltest
    def test_bincount(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, valid_split_axis=0):
                _test_bincount(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

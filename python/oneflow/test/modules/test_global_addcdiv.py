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


@autotest(n=1, check_graph=True)
def _test_addcdiv(test_case, ndim, placement, sbp):
    shape = [random(2, 4) * 8 for i in range(ndim)]
    input = random_tensor(ndim, *shape).to_global(placement=placement, sbp=sbp)
    tensor1 = random_tensor(ndim, *shape).to_global(placement=placement, sbp=sbp)
    tensor2 = random_tensor(ndim, *shape).to_global(placement=placement, sbp=sbp)
    value = random(2, 4).to(int)
    output = torch.addcdiv(input, tensor1, tensor2, value=value)
    return output


class TestModule(flow.unittest.TestCase):
    @globaltest
    def test_addcdiv(test_case):
        ndim = random(2, 4).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_addcdiv(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()

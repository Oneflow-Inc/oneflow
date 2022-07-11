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
def _test_mv(test_case, placement, sbp):
    dim = random(1, 6)
    mat = random_tensor(2, dim1=dim).to_global(placement=placement, sbp=sbp)
    vec = random_tensor(1, dim0=dim).to_global(placement=placement, sbp=sbp)
    return torch.mv(mat, vec)


class TestMvModule(flow.unittest.TestCase):
    @globaltest
    def test_mv(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement):
                _test_mv(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

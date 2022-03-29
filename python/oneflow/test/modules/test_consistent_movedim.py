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
def _test_movedim(test_case, placement, sbp):
    x = random_tensor(
        ndim=4,
        dim1=random(1, 3) * 8,
        dim2=random(1, 3) * 8,
        dim3=random(1, 3) * 8,
        dim4=random(1, 3) * 8,
    ).to_global(placement, sbp)
    z = torch.movedim(x, (0, 1), (2, 3))
    return z


class TestMovedim(flow.unittest.TestCase):
    @globaltest
    def test_movedim(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement):
                _test_movedim(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

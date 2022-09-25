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


@autotest(n=1)
def _test_linalg_cross(test_case, index_size_equal_3, ndim, placement, sbp):
    shape = [random(1, 4).to(int) * 8 for i in range(ndim)]
    shape[index_size_equal_3] = 3
    x = random_tensor(ndim, *shape)
    x = x.to_global(placement=placement, sbp=sbp)
    y = random_tensor(ndim, *shape)
    y = y.to_global(placement=placement, sbp=sbp)
    return torch.cross(
        x, y, dim=index_size_equal_3
    )  # TODO(peihong): will convert to torch.linalg.cross when PyTorch in ci is upgraded to 1.11


class TestLinalgCrossGlobal(flow.unittest.TestCase):
    @globaltest
    def test_linalg_cross(test_case):
        ndim = random(2, 5).to(int).value()
        index_size_equal_3 = random(0, ndim).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(
                placement,
                max_dim=ndim,
                valid_split_axis=[i for i in range(ndim) if i != index_size_equal_3],
            ):
                _test_linalg_cross(test_case, index_size_equal_3, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()

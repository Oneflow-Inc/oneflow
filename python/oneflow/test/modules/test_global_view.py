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

from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@autotest(n=1, check_graph=True)
def _test_global_view(test_case, placement, sbp):
    x = random_tensor(ndim=2, dim0=8, dim1=32).to_global(placement, sbp)
    y = x.view(8, 8, 2, -1)
    return y


@autotest(n=1, check_graph=True)
def _test_global_view_size(test_case, placement, sbp):
    x = random_tensor(ndim=2, dim0=8, dim1=32).to_global(placement, sbp)
    shape = torch.Size([8, 8, 2, -1])
    y = x.view(shape)
    return y


class TestGlobalView(flow.unittest.TestCase):
    @globaltest
    def test_global_view(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_view(test_case, placement, sbp)

    @globaltest
    def test_global_view_size(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_view_size(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

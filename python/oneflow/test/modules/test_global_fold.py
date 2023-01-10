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
def _test_fold_impl(test_case, placement, sbp):
    ndim = 3
    dims = [random(1, 4).to(int).value() * 8 for i in range(ndim)]
    m = torch.nn.Fold(
        output_size=constant(((dims[2] // 4) * 2, 4 * 2)),
        kernel_size=constant(2),
        dilation=constant(1),
        padding=constant(0),
        stride=constant(2),
    )
    m.train(random())

    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    func_y = torch.nn.functional.fold(
        x,
        output_size=constant(((dims[2] // 4) * 2, 4 * 2)),
        kernel_size=constant(2),
        dilation=constant(1),
        padding=constant(0),
        stride=constant(2),
    )
    return y, func_y


class TestFold(flow.unittest.TestCase):
    @globaltest
    def test_fold(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=3):
                _test_fold_impl(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

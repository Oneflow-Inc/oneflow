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


@autotest(n=2, check_graph=True)
def _test_global_tril_without_diag(test_case, placement, sbp):
    x = random_tensor(
        ndim=4,
        dim0=random(1, 3).to(int) * 8,
        dim1=random(1, 3).to(int) * 8,
        dim2=random(1, 3).to(int) * 8,
        dim3=random(1, 3).to(int) * 8,
    ).to_global(placement, sbp)
    y = torch.tril(x)
    y = torch.exp(y)

    return y


@autotest(n=2, check_graph=True)
def _test_global_tril_with_diag(test_case, placement, sbp):
    diagonal = random(-3, 3).to(int)
    x = random_tensor(
        ndim=4,
        dim0=random(1, 4).to(int) * 8,
        dim1=random(1, 4).to(int) * 8,
        dim2=random(1, 4).to(int) * 8,
        dim3=random(1, 4).to(int) * 8,
    ).to_global(placement, sbp)
    y = torch.tril(x, diagonal)
    y = torch.exp(y)

    return y


class TestGlobalTril(flow.unittest.TestCase):
    @globaltest
    def test_global_tril_without_diag(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_global_tril_without_diag(test_case, placement, sbp)

    @globaltest
    def test_global_tril_with_diag(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_global_tril_with_diag(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

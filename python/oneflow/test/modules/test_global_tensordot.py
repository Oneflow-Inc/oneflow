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


@autotest(n=1, check_graph=True, atol=1e-3)
def _test_global_tensordot_against_pytorch(test_case, ndim, placement, sbp):
    k = random(1, 2) * 8
    tensordot_dim = random(0, ndim + 1).to(int)

    x = random_tensor(ndim=ndim, dim0=k, dim1=k, dim2=k, dim3=k).to_global(
        placement=placement, sbp=sbp
    )
    y = random_tensor(ndim=ndim, dim0=k, dim1=k, dim2=k, dim3=k).to_global(
        placement=placement, sbp=sbp
    )
    z = torch.tensordot(x, y, dims=tensordot_dim)
    return z


class TestTensorDotGlobal(flow.unittest.TestCase):
    @globaltest
    def test_tensordot(test_case):
        for placement in all_placement():
            for ndim in range(1, 4):
                for sbp in all_sbp(placement, max_dim=ndim):
                    _test_global_tensordot_against_pytorch(
                        test_case, ndim, placement, sbp
                    )


if __name__ == "__main__":
    unittest.main()

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

import numpy as np

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

# The rtol is too large caused by the expansion of random tensor range
# of #9534. It should be checked again in the future.
@autotest(n=1, check_graph=True, rtol=5e-1, atol=1e-3)
def _test_einsum_tensor_contraction(test_case, placement, sbp):
    dim0 = random(1, 3) * 8
    dim1 = random(1, 3) * 8
    x = random_tensor(
        ndim=4, dim0=random(1, 3) * 8, dim1=dim0, dim2=dim1, dim3=random(1, 3) * 8,
    )
    y = random_tensor(
        ndim=5,
        dim0=random(1, 3) * 8,
        dim1=random(1, 3) * 8,
        dim2=dim0,
        dim3=random(1, 3) * 8,
        dim4=dim1,
    )
    g_x = x.to_global(placement=placement, sbp=sbp)
    g_y = y.to_global(placement=placement, sbp=sbp)
    z = torch.einsum("pqrs,tuqvr->pstuv", g_x, g_y)
    return z


class TestEinsumGlobal(flow.unittest.TestCase):
    @globaltest
    def test_einsum_tensor_contraction(test_case):
        for placement in all_placement():
            if len(np.array(placement.ranks).shape) > 1 and all(
                dim != 1 for dim in np.array(placement.ranks).shape
            ):
                print(
                    f"[{flow.env.get_rank()}] skip TestEinsumConsistent.test_einsum_tensor_contraction with {placement}"
                )
                continue

            for sbp in all_sbp(placement, max_dim=4):
                _test_einsum_tensor_contraction(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

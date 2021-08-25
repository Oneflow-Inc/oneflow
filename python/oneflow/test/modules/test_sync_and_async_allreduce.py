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


def sync_allreduce(x, placement, np_arr, test_case):
    y = x.to_consistent(
        placement=placement, sbp=flow.sbp.partial_sum
    ).to_consistent(placement=placement, sbp=flow.sbp.broadcast)
    test_case.assertTrue(np.allclose(y.to_local().numpy(), np_arr * 10))


def async_allreduce(x, np_arr, test_case):
    y = flow.F.all_reduce(x)
    test_case.assertTrue(np.allclose(y.numpy(), np_arr * 10))


@flow.unittest.skip_unless_1n4d()
class TestP2bOnGPU(flow.unittest.TestCase):
    def test_p2b(test_case):
        placement = flow.placement("cuda", {0: range(4)})
        np_arr = np.array([1, 2])
        x = flow.Tensor(np_arr * (flow.distributed.get_rank() + 1))
        x = x.to("cuda")

        for i in range(1):
            sync_allreduce(x, placement, np_arr, test_case)
            async_allreduce(x, np_arr, test_case)


if __name__ == "__main__":
    unittest.main()

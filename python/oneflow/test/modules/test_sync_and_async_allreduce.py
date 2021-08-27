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


def sync_allreduce(x):
    y = x.to_consistent(sbp=flow.sbp.broadcast)

def async_allreduce(x):
    y = flow.F.all_reduce(x)


@flow.unittest.skip_unless_1n4d()
class TestP2bOnGPU(flow.unittest.TestCase):
    def test_p2b(test_case):
        placement = flow.placement("cuda", {0: range(4)})
        sync_x = flow.ones((128, 1024, 1024), placement=placement, dtype=flow.int32, sbp=flow.sbp.partial_sum)
        async_x = flow.ones((128 * 2, 1024, 1024), device="cuda", dtype=flow.int32)
        i = 0
        for i in range(5000):
            sync_allreduce(sync_x)
            async_allreduce(async_x)
            if i % 20 == 0:
                print(i)


if __name__ == "__main__":
    unittest.main()

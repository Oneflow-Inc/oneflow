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
import os
import oneflow as flow
import oneflow.unittest


def sync_allreduce(x):
    return x.to_global(sbp=flow.sbp.broadcast)


def async_allreduce(x):
    return flow._C.local_all_reduce(x)


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestP2bOnGPU(flow.unittest.TestCase):
    def test_p2b(test_case):
        placement = flow.placement("cuda", range(4))
        sync_x = flow.ones(
            (128, 1024),
            placement=placement,
            dtype=flow.int32,
            sbp=flow.sbp.partial_sum,
        )
        async_x = flow.ones((128 * 2, 1024), device="cuda", dtype=flow.int32)
        i = 0
        for i in range(500):
            synced_y = sync_allreduce(sync_x)
            asynced_y = async_allreduce(async_x)
            if i % 20 == 0:
                print(i)
        print(synced_y.to_local().numpy())
        print(asynced_y.numpy())


if __name__ == "__main__":
    unittest.main()

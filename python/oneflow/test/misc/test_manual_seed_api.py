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

import os
import unittest

import numpy as np
import oneflow as flow

import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestManualSeedApi(flow.unittest.TestCase):
    def test_cuda_manual_seed_all(test_case):
        flow.cuda.manual_seed_all(20)
        x = flow.randn(2, 4, device="cuda:0")
        y = flow.randn(2, 4, device="cuda:1")
        test_case.assertTrue(np.allclose(x.numpy(), y.numpy()))

    def test_cuda_manual_seed(test_case):
        flow.cuda.manual_seed(30)
        device = flow.device("cuda", flow.cuda.current_device())
        x = flow.randn(2, 4, device=device)
        tensor_list = [flow.zeros((2, 4), dtype=flow.int32) for _ in range(2)]
        flow.comm.all_gather(tensor_list, x)
        test_case.assertTrue(
            np.allclose(tensor_list[0].numpy(), tensor_list[1].numpy())
        )

    def test_manual_seed(test_case):
        flow.manual_seed(40)
        x = flow.randn(2, 4, device="cuda:0")
        y = flow.randn(2, 4, device="cuda:1")
        test_case.assertTrue(np.allclose(x.numpy(), y.numpy()))

    def test_set_get_rng_state(test_case):
        x = flow.ByteTensor(5000)
        flow.set_rng_state(x)
        y = flow.get_rng_state()
        test_case.assertTrue(np.allclose(x.numpy(), y.numpy()))


if __name__ == "__main__":
    unittest.main()

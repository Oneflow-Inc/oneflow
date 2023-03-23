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


@flow.unittest.skip_unless_1n1d()
class TestMluMomentumUpdate(flow.unittest.TestCase):
    def test_mlu_momentum_update(test_case):
        def get_updated_tensors(device):
            params = [
                flow.ones(4, 100, 200, 10).to(device),
                flow.ones(23333).to(device) * 2,
            ]
            params[0].grad = flow.ones_like(params[0]).to(device)
            params[1].grad = flow.ones_like(params[1]).to(device) * 0.5
            optimizer = flow.optim.SGD(params, lr=0.001, momentum=0.9)
            optimizer.step()
            return params

        cpu_tensors = get_updated_tensors("cpu")
        mlu_tensors = get_updated_tensors("mlu")
        for cpu, mlu in zip(cpu_tensors, mlu_tensors):
            test_case.assertTrue(np.allclose(cpu.numpy(), mlu.numpy()))


if __name__ == "__main__":
    unittest.main()

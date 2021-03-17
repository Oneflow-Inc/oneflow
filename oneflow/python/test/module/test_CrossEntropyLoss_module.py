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
import oneflow as flow
import unittest
import numpy as np
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)

g_test_samples = [
    {
        "input": np.array(
            [
                [-0.6980871, 0.4765042, -1.969919, 0.28965086, -0.53548324],
                [-0.26332688, 0.27541, 0.30080616, 0.09914763, 0.53522176],
                [0.7332028, 0.38375184, -0.2831992, -0.9833142, 0.387824],
            ]
        ),
        "target": np.array([3, 3, 4], dtype=np.int32),
        "out": np.array([1.1380, 1.7332, 1.4287], dtype=np.float32),
        "out_sum": np.array([4.2999], dtype=np.float32),
    }
]


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_CrossEntropyLoss(test_case):
        global g_test_samples
        for sample in g_test_samples:
            loss = flow.nn.CrossEntropyLoss(reduction=None)
            input = flow.Tensor(sample["input"], dtype=flow.float32)
            target = flow.Tensor(sample["target"], dtype=flow.int32)
            of_out = loss(input, target)
            assert np.allclose(of_out.numpy(), sample["out"], 1e-4, 1e-4)

            loss_sum = flow.nn.CrossEntropyLoss(reduction="sum")
            of_out_sum = loss_sum(input, target)
            assert np.allclose(of_out_sum.numpy(), sample["out_sum"], 1e-4, 1e-4)


if __name__ == "__main__":
    unittest.main()

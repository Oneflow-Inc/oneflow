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
from automated_test_util import *

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestCrossEntropyLossModule(flow.unittest.TestCase):
    @unittest.skip("nn.CrossEntropyLoss has bug")
    @autotest(n=200)
    def test_CrossEntropyLoss_with_random_data(test_case):
        num_classes = random()
        shape = random_tensor(ndim=random(2, 5), dim1=num_classes).value().shape
        m = torch.nn.CrossEntropyLoss(
            reduction=oneof("none", "sum", "mean", nothing()),
            ignore_index=random(0, num_classes) | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(len(shape), *shape).to(device)
        target = random_pytorch_tensor(
            len(shape) - 1, *shape[:1] + shape[2:], low=0, high=num_classes, dtype=int
        ).to(device)
        y = m(x, target)
        return y


if __name__ == "__main__":
    unittest.main()

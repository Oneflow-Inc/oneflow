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
from automated_test_util import *
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestCrossEntropyLossModule(flow.unittest.TestCase):
    @autotest()
    def test_cross_entropy_loss_with_random_data(test_case):
        num_classes = random(low=2)
        device = random_device()
        batch_size = random(low=10, high=100)

        x = random_pytorch_tensor(
            ndim=random(3, 5), dim0=batch_size, dim1=num_classes
        ).to(device)
        target = random_pytorch_tensor(
            2, batch_size, 1, low=0, high=num_classes, dtype=int
        ).to(device)

        ignore_index = (
            random(0, num_classes) | nothing() if num_classes.value() > 2 else nothing()
        )
        m = torch.nn.CrossEntropyLoss(
            reduction=oneof("none", "sum", "mean", nothing()),
            ignore_index=ignore_index,
        )
        m.train(random())
        m.to(device)

        y = m(x, target)
        return y


@flow.unittest.skip_unless_1n1d()
class TestL1LossModule(flow.unittest.TestCase):
    @autotest()
    def test_l1_loss_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().value().shape

        x = random_pytorch_tensor(len(shape), *shape).to(device)
        target = random_pytorch_tensor(len(shape), *shape).to(device)

        m = torch.nn.L1Loss(reduction=oneof("none", "sum", "mean", nothing()))
        m.train(random())
        m.to(device)

        y = m(x, target)
        return y


@flow.unittest.skip_unless_1n1d()
class TestSmoothL1LossModule(flow.unittest.TestCase):
    @autotest()
    def test_smooth_l1_loss_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().value().shape

        x = random_pytorch_tensor(len(shape), *shape).to(device)
        target = random_pytorch_tensor(len(shape), *shape).to(device)

        m = torch.nn.SmoothL1Loss(
            reduction=oneof("none", "sum", "mean", nothing()), beta=oneof(0, 0.5, 1)
        )
        m.train(random())
        m.to(device)

        y = m(x, target)
        return y


if __name__ == "__main__":
    unittest.main()

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
from oneflow.nn.parallel import DistributedDataParallel as ddp
import oneflow.unittest
import numpy as np
import os

train_x = [
    flow.tensor([[1, 2], [2, 3]], dtype=flow.float32),
    flow.tensor([[4, 6], [3, 1]], dtype=flow.float32),
]
train_float32 = [
    flow.tensor([[8], [13]], dtype=flow.float32),
    flow.tensor([[26], [9]], dtype=flow.float32),
]

train_int32 = [
    flow.tensor([[8], [13]], dtype=flow.int32),
    flow.tensor([[26], [9]], dtype=flow.int32),
]


class Model(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = 0.01
        self.iter_count = 100
        self.w1 = flow.nn.Parameter(flow.tensor([[0], [0]],
                                                dtype=flow.float32))

    def forward(self, x, label):
        x1 = flow.matmul(x, self.w1)
        return x1, label


def train(train_x, train_y, test_case):
    m = Model().to("cuda")
    m = ddp(m)
    loss = flow.nn.MSELoss(reduction="sum")
    optimizer = flow.optim.SGD(m.parameters(), m.lr)

    for i in range(0, m.iter_count):
        rank = flow.env.get_rank()
        x = train_x[rank].to("cuda")
        y = train_y[rank].to("cuda")

        y_pred, y2 = m(x, y)
        test_case.assertFalse(y2.requires_grad)
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestDdpMultmpleOutputs(flow.unittest.TestCase):
    def test_outputs_float32(test_case):
        train(train_x, train_float32, test_case)

    def test_outputs_int32(test_case):
        train(train_x, train_int32, test_case)


if __name__ == "__main__":
    unittest.main()

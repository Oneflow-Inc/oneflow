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
import os
import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp
import oneflow.unittest
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict

train_x = [
    flow.tensor([[1, 2], [2, 3]], dtype=flow.float32),
    flow.tensor([[4, 6], [3, 1]], dtype=flow.float32),
]

train_float32 = [
    flow.tensor([[1, 2], [2, 3]], dtype=flow.float32),
    flow.tensor([[4, 6], [3, 1]], dtype=flow.float32),
]

train_int32 = [
    flow.tensor([[8], [13]], dtype=flow.int32),
    flow.tensor([[26], [9]], dtype=flow.int32),
]


class Model(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = 0.01
        self.iter_count = 10
        self.w1 = flow.nn.Parameter(flow.tensor([[0], [0]], dtype=flow.float32))
        self.w2 = flow.nn.Parameter(flow.tensor([[0], [0]], dtype=flow.float32))

    def forward(self, x, label):
        if flow.env.get_rank() == 0:
            x1 = flow.matmul(x, self.w1)
        else:
            x1 = flow.matmul(x, self.w2)
        return ([x1, label + 1], label + 2)


def train(test_case, train_x, device, output, requires_grad):
    m = Model().to(device)
    m = ddp(m)
    loss = flow.nn.MSELoss(reduction="sum")
    optimizer = flow.optim.SGD(m.parameters(), m.lr)

    for i in range(0, m.iter_count):
        rank = flow.env.get_rank()

        x = train_x[rank].clone().to(device)
        y = output[rank].clone().to(device)
        y.requires_grad = requires_grad
        (y_pred, y_add_1), y_add_2 = m(x, y)
        test_case.assertEqual(y_add_1.requires_grad, y.requires_grad)
        test_case.assertEqual(y_add_2.requires_grad, y.requires_grad)
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()


test_device = ["cpu"] if os.getenv("ONEFLOW_TEST_CPU_ONLY") else ["cpu", "cuda"]


@flow.unittest.skip_unless_1n2d()
class TestDdpMultmpleOutputs(flow.unittest.TestCase):
    def test_outputs_float32(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = test_device
        arg_dict["output"] = [train_float32]
        arg_dict["requires_grad"] = [True, False]
        for arg in GenArgDict(arg_dict):
            train(test_case, train_x, **arg)

    def test_outputs_int32(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = test_device
        arg_dict["output"] = [train_int32]
        arg_dict["requires_grad"] = [False]
        for arg in GenArgDict(arg_dict):
            train(test_case, train_x, **arg)


if __name__ == "__main__":
    unittest.main()

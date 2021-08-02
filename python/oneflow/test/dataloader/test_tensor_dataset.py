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
import oneflow.nn as nn
import oneflow.unittest
import oneflow.optim as optim


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


@unittest.skip("optimizer has a bug with 0-dim tensor")
class TestTensorDataset(flow.unittest.TestCase):
    def test_tensor_dataset(test_case):
        num_inputs = 2
        num_examples = 1000
        true_w = [2, -3.4]
        true_b = 4.2
        net = LinearNet(num_inputs)
        flow.nn.init.normal_(net.linear.weight, mean=0, std=0.01)
        flow.nn.init.constant_(net.linear.bias, val=0)
        loss = nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.03)

        features = flow.tensor(
            np.random.normal(0, 1, (num_examples, num_inputs)), dtype=flow.float
        )
        labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
        labels += flow.tensor(
            np.random.normal(0, 0.01, size=labels.size()), dtype=flow.float
        )

        batch_size = 10
        dataset = flow.utils.data.TensorDataset(features, labels)
        data_iter = flow.utils.data.DataLoader(
            dataset, batch_size, shuffle=True, num_workers=0
        )
        num_epochs = 10
        for epoch in range(1, num_epochs + 1):
            for (X, y) in data_iter:
                output = net(X)
                l = loss(output, y).sum()
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            if epoch == num_epochs:
                test_case.assertLess(l.numpy(), 0.00025)


if __name__ == "__main__":
    unittest.main()

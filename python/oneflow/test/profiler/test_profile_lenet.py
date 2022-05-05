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
import oneflow.unittest
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import oneflow.profiler


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class TestProfileLenet(flow.unittest.TestCase):
    def test_lenet(test_case):
        x = flow.randn(2, 3, 32, 32)
        lenet = LeNet()
        # warm up
        for _ in range(10):
            res = lenet(x)
        with oneflow.profiler.profile() as prof:
            with oneflow.profiler.record_function("lenet_forward_total_time") as f:
                for _ in range(2):
                    eager_res = lenet(x)
            with oneflow.profiler.record_function("lenet_backward_total_time") as f:
                eager_res.sum().backward()
        events_avg = prof.key_averages()
        test_case.assertEqual(len(events_avg), 42)
        test_case.assertEqual(events_avg[0].name, "lenet_forward_total_time")
        test_case.assertEqual(events_avg[1].count, 2)
        test_case.assertEqual(events_avg[17].name, "lenet_backward_total_time")
        test_case.assertEqual(events_avg[18].count, 1)
        test_case.assertEqual(events_avg[0].input_shapes, "-")
        test_case.assertNotEqual(events_avg[1].input_shapes, "-")


if __name__ == "__main__":
    unittest.main()

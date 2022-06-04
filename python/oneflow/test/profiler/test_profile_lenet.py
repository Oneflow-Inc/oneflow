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
from tkinter import TRUE
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


def get_event(events, name: str, input_shapes: str = "-"):
    for item in events:
        if item.name == name and item.input_shapes == input_shapes:
            return item
    return None


def _test_lenet(
    test_case,
    on_cuda: bool,
    record_shapes: bool,
    record_bandwidth_for_cuda: bool = False,
):
    x = flow.randn(2, 3, 32, 32)
    lenet = LeNet()
    if on_cuda:
        x = x.to("cuda")
        lenet = lenet.to("cuda")
    activities = [oneflow.profiler.ProfilerActivity.CPU]
    if on_cuda:
        activities.append(oneflow.profiler.ProfilerActivity.CUDA)
    with oneflow.profiler.profile(
        activities=activities,
        record_shapes=record_shapes,
        record_bandwidth_for_cuda=record_bandwidth_for_cuda,
    ) as prof:
        with oneflow.profiler.record_function("lenet_forward_total_time") as f:
            for _ in range(2):
                eager_res = lenet(x)
        with oneflow.profiler.record_function("lenet_backward_total_time") as f:
            eager_res.sum().backward()
    events = prof.key_averages()

    conv_event = get_event(
        events, "conv2d", "[(2,3,32,32), (6,3,5,5)]" if record_shapes else "-"
    )
    test_case.assertIsNotNone(conv_event)
    test_case.assertEqual(conv_event.on_gpu, True if on_cuda else False)

    if on_cuda:
        test_case.assertGreater(conv_event.cpu_time, 0.0)
        test_case.assertGreater(conv_event.cpu_time_total, 0.0)
        test_case.assertGreater(conv_event.gpu_time, 0.0)
        test_case.assertGreater(conv_event.gpu_time_total, 0.0)
    else:
        test_case.assertGreater(conv_event.cpu_time, 0.0)
        test_case.assertGreater(conv_event.cpu_time_total, 0.0)

    test_case.assertEqual(conv_event.count, 2 if record_shapes else 4)
    if record_bandwidth_for_cuda and on_cuda:
        test_case.assertNotEqual(conv_event.bandwidth, -1)

    relu_grad_event = get_event(
        events, "relu_grad", "[(2,6,28,28), (2,6,28,28)]" if record_shapes else "-"
    )
    test_case.assertIsNotNone(relu_grad_event)
    test_case.assertEqual(conv_event.on_gpu, True if on_cuda else False)

    if on_cuda:
        test_case.assertGreater(relu_grad_event.cpu_time, 0.0)
        test_case.assertGreater(relu_grad_event.cpu_time_total, 0.0)
        test_case.assertGreater(relu_grad_event.gpu_time, 0.0)
        test_case.assertGreater(relu_grad_event.gpu_time_total, 0.0)
    else:
        test_case.assertGreater(relu_grad_event.cpu_time, 0.0)
        test_case.assertGreater(relu_grad_event.cpu_time_total, 0.0)

    test_case.assertEqual(relu_grad_event.count, 1 if record_shapes else 4)
    if record_bandwidth_for_cuda and on_cuda:
        test_case.assertNotEqual(relu_grad_event.bandwidth, -1)

    test_case.assertIsNotNone(get_event(events, "lenet_forward_total_time"))
    test_case.assertIsNotNone(get_event(events, "lenet_backward_total_time"))


class TestProfileLenet(flow.unittest.TestCase):
    def test_lenet_cpu(test_case):
        _test_lenet(test_case, on_cuda=False, record_shapes=True)
        _test_lenet(test_case, on_cuda=False, record_shapes=False)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_lenet_cuda(test_case):
        _test_lenet(
            test_case, on_cuda=True, record_shapes=True, record_bandwidth_for_cuda=False
        )
        _test_lenet(
            test_case,
            on_cuda=True,
            record_shapes=False,
            record_bandwidth_for_cuda=False,
        )
        _test_lenet(
            test_case, on_cuda=True, record_shapes=True, record_bandwidth_for_cuda=True
        )
        _test_lenet(
            test_case, on_cuda=True, record_shapes=False, record_bandwidth_for_cuda=True
        )


if __name__ == "__main__":
    unittest.main()

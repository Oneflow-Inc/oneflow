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
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestConv3DModule(flow.unittest.TestCase):
    @autotest(n=10)
    def test_conv3d_with_random_data(test_case):
        channels = random(1, 6)
        m = torch.nn.Conv3d(
            in_channels=channels,
            out_channels=random(1, 6),
            kernel_size=random(1, 3),
            stride=random() | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(1, 5) | nothing(),
            groups=random(1, 5) | nothing(),
            padding_mode=constant("zeros") | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(ndim=5, dim0=2, dim1=channels).to(device)
        y = m(x)
        return y


if __name__ == "__main__":
    unittest.main()

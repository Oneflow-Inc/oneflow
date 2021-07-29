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
from automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestConv3DModule(flow.unittest.TestCase):
    @autotest(n=2)
    def test_against_pytorch(test_case):
        channels = random(1, 6)
        device = random_device()
        m = torch.nn.Conv3d(
            channels,
            random(1, 6),
            random(1, 6),
            stride=random(1, 3) | nothing(),
            padding=random(1, 3) | nothing(),
            dilation=random(1, 3) | nothing(),
            groups=random(1, 3) | nothing(),
            bias=random() | nothing(),
            padding_mode=constant("zeros") | nothing(),
        )
        m.train(random())
        m.to(device)
        x = random_pytorch_tensor(
            ndim=5, dim1=channels, dim2=random(1, 8), dim3=random(1, 8)
        ).to(device)
        y = m(x)
        return y


if __name__ == "__main__":
    unittest.main()

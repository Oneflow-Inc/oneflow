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
from automated_test_util import *

@flow.unittest.skip_unless_1n1d()
class TestMaxPooling(flow.unittest.TestCase):
    @autotest()
    def test_maxpool1d_with_random_data(test_case):
        m = torch.nn.MaxPool1d(
            kernel_size=random(2, 3).to(int),
            stride=random(1, 3).to(int),
            padding=random(0, 1).to(int),
            dilation=random(1, 2).to(int),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(ndim=3, dim0=random(25, 46), dim1=random(33, 55), dim2=random(1, 3)).to(device)
        y = m(x)
        return y

    @autotest()
    def test_maxpool2d_with_random_data(test_case):
        m = torch.nn.MaxPool2d(
            kernel_size=random(2, 3).to(int),
            stride=random(1, 3).to(int),
            padding=random(0, 1).to(int),
            dilation=random(1, 2).to(int),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(ndim=4, dim0=random(2, 4), dim1=random(3, 5), dim2=random(25, 46),
                                  dim3=random(25, 46)).to(device)
        y = m(x)
        return y

    @autotest()
    def test_maxpool3d_with_random_data(test_case):
        m = torch.nn.MaxPool3d(
            kernel_size=random(2, 3).to(int),
            stride=random(1, 3).to(int),
            padding=random(0, 1).to(int),
            dilation=random(1, 2).to(int),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(ndim=5, dim0=random(2, 4), dim1=random(3, 5), dim2=random(5, 6),
                                  dim3=random(25, 46), dim4=random(25, 46)).to(device)
        y = m(x)
        return y


if __name__ == "__main__":
    unittest.main()

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
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t


def unpack_indices(dual_object):
    length = dual_object.__len__().pytorch
    return [dual_object[i] for i in range(length)]


@flow.unittest.skip_unless_1n1d()
class TestMaxPooling(flow.unittest.TestCase):
    @autotest(n=20, auto_backward=True, rtol=1e-4, atol=1e-4)
    def test_maxpool1d_with_random_data(test_case):
        m = torch.nn.MaxPool1d(
            kernel_size=random(4, 6).to(_size_1_t),
            stride=random(1, 3).to(_size_1_t) | nothing(),
            padding=random(1, 3).to(_size_1_t) | nothing(),
            dilation=random(2, 4).to(_size_1_t) | nothing(),
            ceil_mode=random() | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(ndim=3, dim2=random(20, 22)).to(device)
        y = m(x)

        return y

    @autotest(n=20, auto_backward=True, rtol=1e-4, atol=1e-4)
    def test_maxpool2d_with_random_data(test_case):
        m = torch.nn.MaxPool2d(
            kernel_size=random(4, 6).to(_size_2_t),
            stride=random(1, 3).to(_size_2_t) | nothing(),
            padding=random(1, 3).to(_size_2_t) | nothing(),
            dilation=random(2, 4).to(_size_2_t) | nothing(),
            ceil_mode=random() | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(ndim=4, dim2=random(20, 22), dim3=random(20, 22)).to(
            device
        )
        y = m(x)

        return y

    @autotest(n=20, auto_backward=True, rtol=1e-4, atol=1e-4)
    def test_maxpool3d_with_random_data(test_case):
        m = torch.nn.MaxPool3d(
            kernel_size=random(4, 6).to(_size_3_t),
            stride=random(1, 3).to(_size_3_t) | nothing(),
            padding=random(1, 3).to(_size_3_t) | nothing(),
            dilation=random(2, 4).to(_size_3_t) | nothing(),
            ceil_mode=random() | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(
            ndim=5, dim2=random(20, 22), dim3=random(20, 22), dim4=random(20, 22)
        ).to(device)
        y = m(x)

        return y


if __name__ == "__main__":
    unittest.main()

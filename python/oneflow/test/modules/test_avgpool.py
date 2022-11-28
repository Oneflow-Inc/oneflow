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
from oneflow.test_utils.automated_test_util.generators import constant, random_bool
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestAvgPoolingModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_avgpool1d_with_random_data(test_case):
        m = torch.nn.AvgPool1d(
            kernel_size=random(4, 6),
            stride=random(1, 3) | nothing(),
            padding=random(1, 3) | nothing(),
            ceil_mode=random(),
            count_include_pad=random(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=3, dim2=random(20, 22)).to(device)
        y = m(x)
        return y

    @autotest(n=5)
    def test_avgpool2d_with_random_data(test_case):
        m = torch.nn.AvgPool2d(
            kernel_size=random(4, 6),
            stride=random(1, 3) | nothing(),
            padding=random(1, 3) | nothing(),
            ceil_mode=random(),
            count_include_pad=random(),
            divisor_override=random().to(int),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=4, dim2=random(20, 22), dim3=random(20, 22)).to(device)
        y = m(x)
        return y

    # TODO:(zhaoluyang) this test case has probability to fail in backward
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @autotest(n=5, rtol=0.001, atol=0.001, auto_backward=False)
    def test_avgpool2d_with_half_data(test_case):
        m = torch.nn.AvgPool2d(
            kernel_size=random(4, 6),
            stride=random(1, 3) | nothing(),
            padding=random(1, 3) | nothing(),
            ceil_mode=random(),
            count_include_pad=random(),
            divisor_override=random().to(int),
        )
        m.train(random())
        device = gpu_device()
        m.to(device)
        x = (
            random_tensor(
                ndim=4, dim2=random(20, 22), dim3=random(20, 22), requires_grad=False
            )
            .to(device)
            .to(torch.float16)
        )
        y = m(x)
        return y

    @autotest(n=5)
    def test_avgpool3d_with_random_data(test_case):
        m = torch.nn.AvgPool3d(
            kernel_size=random(4, 6),
            stride=random(1, 3) | nothing(),
            padding=random(1, 3) | nothing(),
            ceil_mode=random(),
            count_include_pad=random(),
            divisor_override=random().to(int),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(
            ndim=5, dim2=random(20, 22), dim3=random(20, 22), dim4=random(20, 22)
        ).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestAvgPoolingFunctional(flow.unittest.TestCase):
    @autotest(n=5)
    def test_avgpool1d_functional(test_case):
        device = random_device()
        x = random_tensor(ndim=3, dim2=random(20, 22)).to(device)
        y = torch.nn.functional.avg_pool1d(
            x,
            kernel_size=random(1, 6).to(int),
            stride=random(1, 3).to(int) | nothing(),
            padding=random(1, 3).to(int),
            ceil_mode=random_bool(),
            count_include_pad=random_bool(),
        )
        return y

    @autotest(n=5)
    def test_avgpool2d_functional(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dim2=random(20, 22), dim3=random(20, 22)).to(device)
        y = torch.nn.functional.avg_pool2d(
            x,
            kernel_size=random(1, 6).to(int),
            stride=random(1, 3).to(int) | nothing(),
            padding=random(1, 3).to(int),
            ceil_mode=random_bool(),
            count_include_pad=random_bool(),
        )
        return y

    @autotest(n=5)
    def test_avgpool3d_functional(test_case):
        device = random_device()
        x = random_tensor(
            ndim=5, dim2=random(20, 22), dim3=random(20, 22), dim4=random(20, 22)
        ).to(device)
        y = torch.nn.functional.avg_pool3d(
            x,
            kernel_size=random(1, 6).to(int),
            stride=random(1, 3).to(int) | nothing(),
            padding=random(1, 3).to(int),
            ceil_mode=random_bool(),
            count_include_pad=random_bool(),
        )
        return y

    @profile(torch.nn.functional.avg_pool2d)
    def profile_avgpool2d(test_case):
        torch.nn.functional.avg_pool2d(
            torch.ones(1, 128, 28, 28), kernel_size=3, padding=1
        )
        torch.nn.functional.avg_pool2d(
            torch.ones(1, 128, 28, 28), kernel_size=3, stride=2, padding=1
        )
        torch.nn.functional.avg_pool2d(
            torch.ones(16, 128, 28, 28), kernel_size=3, padding=1
        )
        torch.nn.functional.avg_pool2d(
            torch.ones(16, 128, 28, 28), kernel_size=3, stride=2, padding=1
        )
        torch.nn.functional.avg_pool2d(
            torch.ones(16, 128, 28, 28),
            kernel_size=3,
            stride=2,
            padding=1,
            ceil_mode=True,
        )


if __name__ == "__main__":
    unittest.main()

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
from oneflow.nn.common_types import _size_2_t


@flow.unittest.skip_unless_1n1d()
class TestFold(flow.unittest.TestCase):
    @autotest(n=3, auto_backward=True, rtol=1e-4, atol=1e-4)
    def test_fold_with_random_data_1(test_case):
        m = torch.nn.Fold(
            output_size=constant((4, 4)),
            kernel_size=constant(3),
            dilation=constant(1),
            padding=constant(1),
            stride=constant(1),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(
            ndim=3, dim0=constant(2), dim1=constant(36), dim2=constant(16)
        ).to(device)
        y = m(x)
        func_y = torch.nn.functional.fold(
            x,
            output_size=constant((4, 4)),
            kernel_size=constant(3),
            dilation=constant(1),
            padding=constant(1),
            stride=constant(1),
        )
        return y, func_y

    @autotest(n=3, auto_backward=True, rtol=1e-4, atol=1e-4)
    def test_fold_with_random_data_2(test_case):
        m = torch.nn.Fold(
            output_size=constant((4, 4)),
            kernel_size=constant(3),
            dilation=constant(1),
            padding=constant(0),
            stride=constant(1),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(
            ndim=3, dim0=constant(2), dim1=constant(36), dim2=constant(4)
        ).to(device)
        y = m(x)
        func_y = torch.nn.functional.fold(
            x,
            output_size=constant((4, 4)),
            kernel_size=constant(3),
            dilation=constant(1),
            padding=constant(0),
            stride=constant(1),
        )
        return y, func_y

    @autotest(n=3, auto_backward=True, rtol=1e-4, atol=1e-4)
    def test_fold_with_random_data_3(test_case):
        m = torch.nn.Fold(
            output_size=constant((8, 8)),
            kernel_size=constant(3),
            dilation=constant(1),
            padding=constant(1),
            stride=constant(2),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(
            ndim=3, dim0=constant(2), dim1=constant(72), dim2=constant(16)
        ).to(device)
        y = m(x)
        func_y = torch.nn.functional.fold(
            x,
            output_size=constant((8, 8)),
            kernel_size=constant(3),
            dilation=constant(1),
            padding=constant(1),
            stride=constant(2),
        )
        return y, func_y

    @autotest(n=3, auto_backward=True, rtol=1e-4, atol=1e-4)
    def test_fold_with_random_data_4(test_case):
        m = torch.nn.Fold(
            output_size=constant((8, 8)),
            kernel_size=constant(3),
            dilation=constant(2),
            padding=constant(1),
            stride=constant(2),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(
            ndim=3, dim0=constant(2), dim1=constant(9), dim2=constant(9)
        ).to(device)
        y = m(x)
        func_y = torch.nn.functional.fold(
            x,
            output_size=constant((8, 8)),
            kernel_size=constant(3),
            dilation=constant(2),
            padding=constant(1),
            stride=constant(2),
        )
        return y, func_y

    @profile(torch.nn.functional.fold)
    def profile_fold(test_case):
        x = torch.ones(128, 128, 4)
        torch.nn.functional.fold(x, output_size=(4, 4), kernel_size=(2, 2), stride=2)


if __name__ == "__main__":
    unittest.main()

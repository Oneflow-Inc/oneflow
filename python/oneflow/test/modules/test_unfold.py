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
class TestUnfold(flow.unittest.TestCase):
    @autotest(n=50, auto_backward=True, rtol=1e-4, atol=1e-4)
    def test_unfold_with_random_data(test_case):
        m = torch.nn.Unfold(
            kernel_size=random(1, 3).to(_size_2_t),
            dilation=random(1, 2).to(_size_2_t) | nothing(),
            padding=random(0, 1).to(_size_2_t) | nothing(),
            stride=random(1, 2).to(_size_2_t) | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(
            ndim=4,
            dim0=random(1, 5),
            dim1=random(1, 5),
            dim2=random(10, 20),
            dim3=random(10, 20),
        ).to(device)
        y = m(x)
        func_y = torch.nn.functional.unfold(
            x,
            kernel_size=random(1, 3).to(_size_2_t),
            dilation=random(1, 2).to(_size_2_t) | nothing(),
            padding=random(0, 1).to(_size_2_t) | nothing(),
            stride=random(1, 2).to(_size_2_t) | nothing(),
        )
        return y, func_y


if __name__ == "__main__":
    unittest.main()

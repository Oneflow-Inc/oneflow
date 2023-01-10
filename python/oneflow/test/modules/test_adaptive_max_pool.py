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
from oneflow.nn.common_types import _size_1_t
from packaging import version
import torch as torch_original
from typing import Union, Tuple
import numpy as np


from oneflow.test_utils.automated_test_util import *

NoneType = type(None)


@flow.unittest.skip_unless_1n1d()
class TestAdaptiveMaxPool(flow.unittest.TestCase):
    @autotest(n=5)
    def test_adaptive_maxpool1d(test_case):
        m = torch.nn.AdaptiveMaxPool1d(output_size=random().to(_size_1_t))
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=3).to(device)
        y = m(x)
        return y

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_adaptive_maxpool2d_manually(test_case):
        def _test_adaptive_max_pool_nd(input_shape, output_shape, m1, m2):
            input_np = np.random.rand(2, 3, *input_shape)
            input_pt = torch_original.tensor(
                input_np, device="cuda", requires_grad=True
            )
            input_of = flow.tensor(input_np, device="cuda", requires_grad=True)

            m_pt = m1(output_shape, True)
            m_of = m2(output_shape, True)

            output_pt = m_pt(input_pt)
            output_of = m_of(input_of)

            sum_pt = torch_original.sum(output_pt[0])
            sum_of = flow.sum(output_of[0])

            sum_pt.backward()
            sum_of.backward()

            test_case.assertTrue(
                np.array_equal(
                    output_pt[0].detach().cpu().numpy(),
                    output_of[0].detach().cpu().numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(
                    output_pt[1].detach().cpu().numpy(),
                    output_of[1].detach().cpu().numpy(),
                )
            )
            test_case.assertTrue(
                np.array_equal(input_pt.grad.cpu().numpy(), input_of.grad.cpu().numpy())
            )

        _test_adaptive_max_pool_nd(
            (10, 11),
            (3, 4),
            torch_original.nn.AdaptiveMaxPool2d,
            flow.nn.AdaptiveMaxPool2d,
        )
        _test_adaptive_max_pool_nd(
            (10, 11, 12),
            (3, 4, 5),
            torch_original.nn.AdaptiveMaxPool3d,
            flow.nn.AdaptiveMaxPool3d,
        )

    @profile(torch.nn.functional.adaptive_max_pool1d)
    def profile_adaptive_max_pool1d(test_case):
        torch.nn.functional.adaptive_max_pool1d(torch.ones(1, 64, 8), 5)

    @profile(torch.nn.functional.adaptive_max_pool2d)
    def profile_adaptive_max_pool2d(test_case):
        torch.nn.functional.adaptive_max_pool2d(torch.ones(1, 64, 10, 9), 7)
        torch.nn.functional.adaptive_max_pool2d(torch.ones(1, 64, 8, 9), (5, 7))

    @profile(torch.nn.functional.adaptive_max_pool3d)
    def profile_adaptive_max_pool3d(test_case):
        torch.nn.functional.adaptive_max_pool3d(torch.ones(1, 64, 8, 9, 10), (5, 7, 9))
        torch.nn.functional.adaptive_max_pool3d(torch.ones(1, 64, 10, 9, 8), 7)


if __name__ == "__main__":
    unittest.main()

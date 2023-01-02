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
from collections import OrderedDict
import numpy as np

from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestLinspace(flow.unittest.TestCase):
    @autotest(n=5, auto_backward=False, rtol=1e-5, atol=1e-5, check_graph=True)
    def test_linspace_int_with_random_data(test_case):
        start = random().to(int)
        end = start + random().to(int)
        steps = random(0, end - start).to(int)
        x = torch.linspace(start=start, end=end, steps=steps)
        device = random_device()
        x.to(device)
        return x

    @autotest(n=5, auto_backward=False, rtol=1e-5, atol=1e-5, check_graph=True)
    def test_linspace_float_with_random_data(test_case):
        start = random()
        end = start + random()
        steps = random(0, end - start).to(int)
        x = torch.linspace(start=start, end=end, steps=steps)
        device = random_device()
        x.to(device)
        return x

    @autotest(n=5, auto_backward=False)
    def test_linspace_with_scalar_tensor_as_params(test_case):
        start = random_tensor(2, 3, 4, requires_grad=False).mean()
        end = start + random_tensor(2, 3, 4, requires_grad=False).mean()
        steps = random(0, 10).to(int)
        y = torch.linspace(start=start, end=end, steps=steps)
        return y

    def test_global_naive(test_case):
        placement = flow.placement("cpu", ranks=[0])
        sbp = (flow.sbp.broadcast,)
        x = flow.linspace(start=0, end=10, steps=2, placement=placement, sbp=sbp)
        test_case.assertEqual(x.sbp, sbp)
        test_case.assertEqual(x.placement, placement)

    def test_linspace_in_transformer_bug(test_case):
        drop_path_rate = 0.1
        depths = [2, 2, 6, 2]
        flow_res = flow.linspace(0, drop_path_rate, sum(depths))
        torch_res = np.array(
            [
                0.0000,
                0.0091,
                0.0182,
                0.0273,
                0.0364,
                0.0455,
                0.0545,
                0.0636,
                0.0727,
                0.0818,
                0.0909,
                0.1000,
            ]
        )
        test_case.assertTrue(np.allclose(flow_res.numpy(), torch_res, atol=1e-4))
        drop_path_rate = 0.2
        depths = [2, 2, 6, 2]
        flow_res = flow.linspace(0, drop_path_rate, sum(depths))
        torch_res = np.array(
            [
                0.0000,
                0.0182,
                0.0364,
                0.0545,
                0.0727,
                0.0909,
                0.1091,
                0.1273,
                0.1455,
                0.1636,
                0.1818,
                0.2000,
            ]
        )
        test_case.assertTrue(np.allclose(flow_res.numpy(), torch_res, atol=1e-4))
        drop_path_rate = 0.3
        depths = [2, 2, 18, 2]
        flow_res = flow.linspace(0, drop_path_rate, sum(depths))
        torch_res = np.array(
            [
                0.0000,
                0.0130,
                0.0261,
                0.0391,
                0.0522,
                0.0652,
                0.0783,
                0.0913,
                0.1043,
                0.1174,
                0.1304,
                0.1435,
                0.1565,
                0.1696,
                0.1826,
                0.1957,
                0.2087,
                0.2217,
                0.2348,
                0.2478,
                0.2609,
                0.2739,
                0.2870,
                0.3000,
            ]
        )
        test_case.assertTrue(np.allclose(flow_res.numpy(), torch_res, atol=1e-4))
        drop_path_rate = 0.1
        depths = [2, 2, 18, 2]
        flow_res = flow.linspace(0, drop_path_rate, sum(depths))
        torch_res = np.array(
            [
                0.0000,
                0.0043,
                0.0087,
                0.0130,
                0.0174,
                0.0217,
                0.0261,
                0.0304,
                0.0348,
                0.0391,
                0.0435,
                0.0478,
                0.0522,
                0.0565,
                0.0609,
                0.0652,
                0.0696,
                0.0739,
                0.0783,
                0.0826,
                0.0870,
                0.0913,
                0.0957,
                0.1000,
            ]
        )
        test_case.assertTrue(np.allclose(flow_res.numpy(), torch_res, atol=1e-4))
        drop_path_rate = 0.5
        depths = [2, 2, 18, 2]
        flow_res = flow.linspace(0, drop_path_rate, sum(depths))
        torch_res = np.array(
            [
                0.0000,
                0.0217,
                0.0435,
                0.0652,
                0.0870,
                0.1087,
                0.1304,
                0.1522,
                0.1739,
                0.1957,
                0.2174,
                0.2391,
                0.2609,
                0.2826,
                0.3043,
                0.3261,
                0.3478,
                0.3696,
                0.3913,
                0.4130,
                0.4348,
                0.4565,
                0.4783,
                0.5000,
            ]
        )
        test_case.assertTrue(np.allclose(flow_res.numpy(), torch_res, atol=1e-4))

    def test_linspace_start_equal_end_bug(test_case):
        flow_res = flow.linspace(0, 0.0, 12).numpy()
        torch_res = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        test_case.assertTrue(np.allclose(flow_res, torch_res, atol=1e-4))


if __name__ == "__main__":
    unittest.main()

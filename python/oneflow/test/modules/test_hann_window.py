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
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestHannWindow(flow.unittest.TestCase):
    @autotest(n=1, auto_backward=False, check_graph=True)
    def test_hann_window(test_case):
        device = random_device()
        window_length = random(1, 8).to(int).value()
        periodic = random_bool().value()
        output = torch.hann_window(window_length, periodic, device=device)
        return output

    def test_hann_window_global(test_case):
        placement = flow.placement("cpu", ranks=[0])
        sbp = (flow.sbp.broadcast,)
        window_length = random(1, 8).to(int).value()
        periodic = random_bool().value()
        output = flow.hann_window(window_length, periodic, placement=placement, sbp=sbp)
        test_case.assertEqual(output.sbp, sbp)
        test_case.assertEqual(output.placement, placement)

    def test_hann_window_dtype(test_case):
        device = random_device().value()
        window_length = random(1, 8).to(int).value()
        periodic = random_bool().value()
        dtype = flow.float64
        output = flow.hann_window(window_length, periodic, device=device, dtype=dtype)
        test_case.assertEqual(output.dtype, dtype)

    @profile(torch.hann_window)
    def profile_hann_window(test_case):
        torch.hann_window(128000, periodic=True)
        torch.hann_window(128001, periodic=False)


if __name__ == "__main__":
    unittest.main()

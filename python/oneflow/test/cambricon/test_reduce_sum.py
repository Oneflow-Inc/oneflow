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


def _test_reduce_sum_impl(test_case, device="mlu", data_type=flow.float32):
    input = flow.tensor(
        np.random.randn(2, 3) - 0.5, dtype=data_type, device=flow.device(device)
    )
    of_out = flow.sum(input, dim=0)
    np_out = np.sum(input.numpy(), axis=0)

    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-04, 1e-04))
    input = flow.tensor(
        np.random.randn(2, 3), dtype=data_type, device=flow.device(device)
    )
    of_out = flow.sum(input, dim=0)
    np_out = np.sum(input.numpy(), axis=0)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-04, 1e-04))
    input = flow.tensor(
        np.random.randn(2, 3), dtype=data_type, device=flow.device(device)
    )
    of_out = flow.sum(input, dim=1)
    of_out2 = input.sum(dim=1)
    np_out = np.sum(input.numpy(), axis=1)
    test_case.assertTrue(np.allclose(of_out2.numpy(), of_out.numpy(), 1e-04, 1e-04))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-04, 1e-04))
    input = flow.tensor(
        np.random.randn(4, 5, 6) - 0.5,
        dtype=data_type,
        device=flow.device(device),
        requires_grad=False,
    )
    of_out = flow.sum(input, dim=(2, 1))
    np_out = np.sum(input.numpy(), axis=(2, 1))
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-04, 1e-04))


@flow.unittest.skip_unless_1n1d()
class TestReduceSumModule(flow.unittest.TestCase):
    def test_reduce_sum(test_case):
        _test_reduce_sum_impl(test_case)


if __name__ == "__main__":
    unittest.main()

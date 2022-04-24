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

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util.torch_flow_dual_object import autotest


def _test_search_sorted(test_case, input_dtype, device):
    sorted_sequence = flow.tensor(
        np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]),
        dtype=input_dtype,
        device=flow.device(device),
    )
    values = flow.tensor(
        np.array([[3, 6, 9], [3, 6, 9]]), dtype=input_dtype, device=flow.device(device)
    )
    gt = np.array([[1, 3, 4], [1, 2, 4]])
    output = flow.searchsorted(sorted_sequence, values)
    test_case.assertTrue(np.allclose(output.numpy(), gt, 0.0001, 0.0001))
    test_case.assertTrue(output.dtype == flow.int64)


def _test_search_sorted_1(test_case, input_dtype, device):
    sorted_sequence = flow.tensor(
        np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]),
        dtype=input_dtype,
        device=flow.device(device),
    )
    values = flow.tensor(
        np.array([[3, 6, 9], [3, 6, 9]]), dtype=input_dtype, device=flow.device(device)
    )
    gt = np.array([[2, 3, 5], [1, 3, 4]])
    output = flow.searchsorted(sorted_sequence, values, right=True, side="right")
    test_case.assertTrue(np.allclose(output.numpy(), gt, 0.0001, 0.0001))
    test_case.assertTrue(output.dtype == flow.int64)


def _test_search_sorted_2(test_case, input_dtype, device):
    sorted_sequence_1d = flow.tensor(
        np.array([1, 3, 5, 7, 9]), dtype=input_dtype, device=flow.device(device)
    )
    values = flow.tensor(
        np.array([3, 6, 9]), dtype=input_dtype, device=flow.device(device)
    )
    gt = np.array([1, 3, 4])
    output = flow.searchsorted(sorted_sequence_1d, values)
    test_case.assertTrue(np.allclose(output.numpy(), gt, 0.0001, 0.0001))
    test_case.assertTrue(output.dtype == flow.int64)


def _test_search_sorted_3(test_case, input_dtype, device):
    sorted_sequence = flow.tensor(
        np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]),
        dtype=input_dtype,
        device=flow.device(device),
    )
    values = flow.tensor(
        np.array([[3, 6, 9], [3, 6, 9]]), dtype=input_dtype, device=flow.device(device)
    )
    gt = np.array([[1, 3, 4], [1, 2, 4]])
    output = flow.searchsorted(sorted_sequence, values, out_int32=True)
    test_case.assertTrue(np.allclose(output.numpy(), gt, 0.0001, 0.0001))
    test_case.assertTrue(output.dtype == flow.int32)


def _test_search_sorted_4(test_case, input_dtype, device):
    sorted_sequence = flow.tensor(
        np.array([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]]),
        dtype=input_dtype,
        device=flow.device(device),
    )
    values = flow.tensor(
        np.array([[3, 6, 9], [3, 6, 9]]), dtype=input_dtype, device=flow.device(device)
    )
    sorter = flow.tensor(
        np.array([[4, 3, 2, 1, 0], [3, 2, 4, 0, 1]]),
        dtype=flow.int64,
        device=flow.device(device),
    )
    gt = np.array([[0, 5, 5], [0, 0, 2]])
    output = flow.searchsorted(sorted_sequence, values, sorter=sorter)
    test_case.assertTrue(np.allclose(output.numpy(), gt, 0.0001, 0.0001))
    test_case.assertTrue(output.dtype == flow.int64)


def _test_search_sorted_5(test_case, input_dtype, device):
    sorted_sequence_1d = flow.tensor(
        np.array([1, 3, 5, 7, 9]), dtype=input_dtype, device=flow.device(device)
    )
    gt = np.array(2)
    output = flow.searchsorted(sorted_sequence_1d, 5)
    test_case.assertTrue(np.allclose(output.numpy(), gt, 0.0001, 0.0001))
    test_case.assertTrue(output.dtype == flow.int64)


def _test_search_sorted_6(test_case, input_dtype, device):
    sorted_sequence_1d = flow.tensor(
        np.array([1, 3, 5, 7, 9]), dtype=input_dtype, device=flow.device(device)
    )
    gt = np.array(3)
    output = flow.searchsorted(sorted_sequence_1d, 5, right=True, side="right")
    test_case.assertTrue(np.allclose(output.numpy(), gt, 0.0001, 0.0001))
    test_case.assertTrue(output.dtype == flow.int64)


def _test_search_sorted_7(test_case, input_dtype, device):
    sorted_sequence_1d = flow.tensor(
        np.array([1, 3, 5, 7, 9]), dtype=input_dtype, device=flow.device(device)
    )
    gt = np.array(2)
    output = flow.searchsorted(sorted_sequence_1d, 5, out_int32=True)
    test_case.assertTrue(np.allclose(output.numpy(), gt, 0.0001, 0.0001))
    test_case.assertTrue(output.dtype == flow.int32)


@flow.unittest.skip_unless_1n1d()
class TestSearchSorted(flow.unittest.TestCase):
    def test_search_sorted(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_search_sorted,
            _test_search_sorted_1,
            _test_search_sorted_2,
            _test_search_sorted_3,
            _test_search_sorted_4,
            _test_search_sorted_5,
            _test_search_sorted_6,
            _test_search_sorted_7,
        ]
        arg_dict["input_dtype"] = [
            flow.int8,
            flow.int32,
            flow.int64,
            flow.float,
            flow.double,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=20, auto_backward=False, check_dtype=True)
    def test_search_sorted(test_case):
        device = random_device()
        sorted_sequence = random_tensor(ndim=2, dim0=2, dim1=3).to(device)
        values = random_tensor(ndim=2, dim0=2).to(device)
        right = oneof(True, False)
        y = torch.searchsorted(
            sorted_sequence, values, out_int32=oneof(True, False), right=right,
        )
        return y


if __name__ == "__main__":
    unittest.main()

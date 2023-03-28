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
import numpy as np

from oneflow.test_utils.automated_test_util import *


def _get_indexes(device):
    return (
        constant(
            torch.tensor(np.array([[0, 1], [1, 0]]), dtype=torch.int64, device=device)
        ),
        constant(
            torch.tensor(np.array([[1, 0], [0, 1]]), dtype=torch.int64, device=device)
        ),
        constant(
            torch.tensor(np.array([[1, 0], [1, 0]]), dtype=torch.int64, device=device)
        ),
        constant(
            torch.tensor(np.array([[0, 1], [0, 1]]), dtype=torch.int64, device=device)
        ),
    )


def _test_scatter(test_case, test_scalar: bool, dim: int):
    device = random_device()
    input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
    src = 3.14 if test_scalar else random_tensor(ndim=2, dim0=2, dim1=2).to(device)
    y = torch.scatter(input, dim, oneof(*_get_indexes(device)), src)
    return y


def _test_scatter_add(test_case, dim: int):
    device = random_device()
    input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
    src = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
    y = torch.scatter_add(input, dim, oneof(*_get_indexes(device)), src)
    return y


def _test_scatter_reduce(test_case, dim: int):
    device = random_device()
    input = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
    src = random_tensor(ndim=2, dim0=2, dim1=2).to(device)
    y = torch.scatter(
        input,
        dim,
        oneof(*_get_indexes(device)),
        src,
        reduce=oneof("add", "multiply", nothing()),
    )
    return y


@flow.unittest.skip_unless_1n1d()
class TestScatterOpsModule(flow.unittest.TestCase):
    @autotest(n=10)
    def test_scatter_with_random_data(test_case):
        return _test_scatter(test_case, oneof(True, False), oneof(0, 1, -1))

    @autotest(n=5)
    def test_scatter_add_with_random_data(test_case):
        return _test_scatter_add(test_case, oneof(0, 1))

    @autotest(
        n=5, auto_backward=False
    )  # peihong: pytorch dose not support backward when reduce is add or multiply
    def test_scatter_reduce_with_random_data(test_case):
        return _test_scatter_reduce(test_case, oneof(0, 1))


if __name__ == "__main__":
    unittest.main()

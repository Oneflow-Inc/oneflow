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

import oneflow as flow

from oneflow.test_utils.automated_test_util import *


def _test_bitwise_op(test_case, op):
    device = random_device()
    dims_kwargs = {
        "ndim": 4,
        "dim0": random(low=4, high=8).to(int),
        "dim1": random(low=4, high=8).to(int),
        "dim2": random(low=4, high=8).to(int),
        "dim3": random(low=4, high=8).to(int),
    }
    # TODO(WangYi): oneflow doesn't support conversion between uint8 and int8
    # So, use "index" instead of "int" in `random_dtype`
    x_dtype = random_dtype(["index", "bool", "unsigned"])
    y_dtype = random_dtype(["index", "bool", "unsigned"])
    x = random_tensor(dtype=int, **dims_kwargs,).to(device).to(x_dtype)
    y = random_tensor(dtype=int, **dims_kwargs,).to(device).to(y_dtype)
    bool_tensor = random_tensor(low=-1, high=1, **dims_kwargs,).to(device) > 0
    return op(op(x, y), bool_tensor)


def _test_scalar_bitwise(test_case, op, inverse=False):
    # inverse=True for testing declarition Op(Scalar, Tensor)
    device = random_device()
    dtype = random_dtype(["int", "bool", "unsigned"])
    x = (
        random_tensor(
            ndim=4,
            dim0=random(low=4, high=8).to(int),
            dim1=random(low=4, high=8).to(int),
            dim2=random(low=4, high=8).to(int),
            dim3=random(low=4, high=8).to(int),
            dtype=int,
        )
        .to(device)
        .to(dtype)
    )
    scalar = random(low=-10, high=10).to(int)
    bool_scalar = random(low=0, high=2, dtype=int).to(bool)
    if inverse:
        # torch doesn't support bitwise_and(Scalar, Tensor), so manually compare results
        torch_result = torch.bitwise_and(
            torch.bitwise_and(x, scalar), bool_scalar
        ).pytorch
        flow_result = flow.bitwise_and(
            bool_scalar.value(), flow.bitwise_and(x.oneflow, scalar.value())
        )
        test_case.assertTrue(
            np.array_equal(torch_result.cpu().numpy(), flow_result.numpy())
        )
    else:
        result = op(op(x, scalar), bool_scalar)
    return result


@flow.unittest.skip_unless_1n1d()
class TestBitwiseAndModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_and(test_case):
        return _test_bitwise_op(test_case, torch.bitwise_and)

    @autotest(n=20, auto_backward=False)
    def test_scalar_bitwise_and(test_case):
        return _test_scalar_bitwise(
            test_case, torch.bitwise_and, inverse=random_bool().value()
        )


@flow.unittest.skip_unless_1n1d()
class TestBitwiseOrModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_or(test_case):
        return _test_bitwise_op(test_case, torch.bitwise_or)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_or(test_case):
        return _test_scalar_bitwise(
            test_case, torch.bitwise_or, inverse=random_bool().value()
        )


@flow.unittest.skip_unless_1n1d()
class TestBitwiseXorModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_xor(test_case):
        return _test_bitwise_op(test_case, torch.bitwise_xor)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_xor(test_case):
        return _test_scalar_bitwise(
            test_case, torch.bitwise_xor, inverse=random_bool().value()
        )


@flow.unittest.skip_unless_1n1d()
class TestBitwiseNotModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_not(test_case):
        device = random_device()
        # TODO(WangYi): oneflow doesn't support conversion between uint8 and int8
        # So, use "index" instead of "int" in `random_dtype`
        dtype = random_dtype(["index", "bool", "unsigned"])
        x = (
            random_tensor(
                ndim=4,
                dim0=random(low=4, high=8).to(int),
                dim1=random(low=4, high=8).to(int),
                dim2=random(low=4, high=8).to(int),
                dim3=random(low=4, high=8).to(int),
                dtype=int,
                high=10,
            )
            .to(device)
            .to(dtype)
        )
        return torch.bitwise_not(x)


if __name__ == "__main__":
    unittest.main()

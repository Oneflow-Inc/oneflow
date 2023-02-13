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


@flow.unittest.skip_unless_1n1d()
class TestBitwiseAndModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_and(test_case):
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
        return torch.bitwise_and(torch.bitwise_and(x, y), bool_tensor)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_and(test_case):
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
        bool_scalar = random(low=0, high=2).to(bool)
        # torch doesn't support bitwise_and(Scalar, Tensor)
        result = torch.bitwise_and(torch.bitwise_and(x, scalar), bool_scalar)
        return result

    # test declaration for bitwise_and(Scalar, Tensor)
    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_and2(test_case):
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
        bool_scalar = random(low=0, high=2).to(bool)
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


@flow.unittest.skip_unless_1n1d()
class TestBitwiseOrModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_or(test_case):
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
        return torch.bitwise_or(torch.bitwise_or(x, y), bool_tensor)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_or(test_case):
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
        bool_scalar = random(low=0, high=2).to(bool)
        result = torch.bitwise_or(torch.bitwise_or(x, scalar), bool_scalar)
        return result

    # test declaration for bitwise_or(Scalar, Tensor)
    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_or2(test_case):
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
        bool_scalar = random(low=0, high=2).to(bool)
        # torch doesn't support bitwise_or(Scalar, Tensor), so manually compare results
        torch_result = torch.bitwise_or(
            torch.bitwise_or(x, scalar), bool_scalar
        ).pytorch
        flow_result = flow.bitwise_or(
            bool_scalar.value(), flow.bitwise_or(x.oneflow, scalar.value())
        )
        test_case.assertTrue(
            np.array_equal(torch_result.cpu().numpy(), flow_result.numpy())
        )


@flow.unittest.skip_unless_1n1d()
class TestBitwiseXorModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_xor(test_case):
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
        return torch.bitwise_xor(torch.bitwise_xor(x, y), bool_tensor)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_xor(test_case):
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
        bool_scalar = random(low=0, high=2).to(bool)
        result = torch.bitwise_xor(torch.bitwise_xor(x, scalar), bool_scalar)
        return result

    # test declaration for bitwise_xor(Scalar, Tensor)
    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_xor2(test_case):
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
        bool_scalar = random(low=0, high=2).to(bool)
        # torch doesn't support bitwise_xor(Scalar, Tensor), so manually compare results
        torch_result = torch.bitwise_xor(
            torch.bitwise_xor(x, scalar), bool_scalar
        ).pytorch
        flow_result = flow.bitwise_xor(
            bool_scalar.value(), flow.bitwise_xor(x.oneflow, scalar.value())
        )
        test_case.assertTrue(
            np.array_equal(torch_result.cpu().numpy(), flow_result.numpy())
        )

    # def test_scalar_logical_xor(test_case):
    #     arg_dict = OrderedDict()
    #     arg_dict["test_fun"] = [_test_tensor_scalar_logical_xor]
    #     arg_dict["shape"] = [(2, 3), (2, 4, 5)]
    #     arg_dict["scalar"] = [1, 0]
    #     arg_dict["dtype"] = [flow.float32, flow.int32]
    #     arg_dict["device"] = ["cpu", "cuda"]
    #     for arg in GenArgList(arg_dict):
    #         arg[0](test_case, *arg[1:])

    # @autotest(n=10, auto_backward=False, check_graph=True)
    # def test_logical_xor_with_random_data(test_case):
    #     device = random_device()
    #     shape = random_tensor().oneflow.shape
    #     x1 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
    #     x2 = random_tensor(len(shape), *shape, requires_grad=False).to(device)
    #     y = torch.logical_xor(x1, x2)
    #     return y

    # @autotest(n=10, auto_backward=False, check_graph=True)
    # def test_logical_xor_bool_with_random_data(test_case):
    #     device = random_device()
    #     shape = random_tensor().oneflow.shape
    #     x1 = random_tensor(len(shape), *shape, requires_grad=False).to(
    #         device=device, dtype=torch.bool
    #     )
    #     x2 = random_tensor(len(shape), *shape, requires_grad=False).to(
    #         device=device, dtype=torch.bool
    #     )
    #     y = torch.logical_xor(x1, x2)
    #     return y


if __name__ == "__main__":
    unittest.main()

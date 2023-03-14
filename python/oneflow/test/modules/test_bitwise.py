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
    x = random_tensor(dtype=int, **dims_kwargs,low=0,high=2).to(device).to(x_dtype)
    y = random_tensor(dtype=int, **dims_kwargs,low=0,high=2).to(device).to(y_dtype)
    bool_tensor = random_tensor(low=-1, high=1, **dims_kwargs,).to(device) > 0
    return op(op(x, y), bool_tensor)

def _test_bitwise_inplace_op(test_case, op,inplace_op):
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
    dtype = random_dtype(["index", "bool", "unsigned"])
    x = random_tensor(dtype=int, **dims_kwargs,low=0,high=2).to(device).to(dtype)
    y = random_tensor(dtype=int, **dims_kwargs,low=0,high=2).to(device).to(dtype)
    bool_tensor = random_tensor(low=-1, high=1, **dims_kwargs,).to(device) > 0
    result = op(op(x,y),bool_tensor)

    x_flow = x.oneflow.clone()
    y_flow = y.oneflow.clone()
    bool_tensor_flow = bool_tensor.oneflow.clone()
    inplace_op(inplace_op(x_flow,y_flow),bool_tensor_flow)
    test_case.assertTrue(
        np.allclose(
            x_flow.numpy(),
            result.pytorch.cpu().numpy()
        )
    )
    



def _test_scalar_bitwise(test_case, op):
    device = random_device()
    dtype = random_dtype(["int", "bool", "unsigned"])
    x = (
        random_tensor(
            ndim=4,
            dim0=random(low=4, high=8).to(int),
            dim1=random(low=4, high=8).to(int),
            dim2=random(low=4, high=8).to(int),
            dim3=random(low=4, high=8).to(int),
            low=0,
            high=2,
            dtype=int,
        )
        .to(device)
        .to(dtype)
    )
    scalar = random(low=-10, high=10).to(int)
    bool_scalar = random_bool()
    result = op(op(x, scalar), bool_scalar)
    return result



def _test_scalar_bitwise_inplace(test_case, op, inplace_op):
    device = random_device()
    dtype = random_dtype(["int", "bool", "unsigned"])
    x = (
        random_tensor(
            ndim=4,
            dim0=random(low=4, high=8).to(int),
            dim1=random(low=4, high=8).to(int),
            dim2=random(low=4, high=8).to(int),
            dim3=random(low=4, high=8).to(int),
            low=0,
            high=2,
            dtype=int,
        )
        .to(device)
        .to(dtype)
    )
    scalar = random(low=-10, high=10).to(dtype)
    # print(scalar)
    bool_scalar = random_bool()
    # print(bool_scalar)
    result = op(op(x, scalar), bool_scalar)
    
    x_flow = x.oneflow.clone()
    scalar_flow = scalar.to(dtype).value()
    bool_scalar_flow = bool_scalar.value()
    # inplace_op(inplace_op(x_flow,scalar_flow),bool_scalar_flow)
    inplace_op(x_flow,scalar_flow)
    inplace_op(x_flow,bool_scalar_flow)
    test_case.assertTrue(
        np.allclose(
            x_flow.numpy(),
            result.pytorch.cpu().numpy()
        )
    )


# Bitwise ops only accept integral dtype,
# so auto_backward isn't necessary
@flow.unittest.skip_unless_1n1d()
class TestBitwiseAndModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_and(test_case):
        return _test_bitwise_op(test_case, torch.bitwise_and)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_and(test_case):
        return _test_scalar_bitwise(test_case, torch.bitwise_and,)

@flow.unittest.skip_unless_1n1d()
class TestBitwiseAndInplaceModule(flow.unittest.TestCase):
    @autotest(n=3, auto_backward=False)
    def test_bitwise_and_inplace(test_case):
        return _test_bitwise_inplace_op(test_case, torch.bitwise_and,flow.bitwise_and_)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_and_inplace(test_case):
        return _test_scalar_bitwise_inplace(test_case, torch.bitwise_and,flow.bitwise_and_)

@flow.unittest.skip_unless_1n1d()
class TestBitwiseOrModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_or(test_case):
        return _test_bitwise_op(test_case, torch.bitwise_or)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_or(test_case):
        return _test_scalar_bitwise(test_case, torch.bitwise_or,)

@flow.unittest.skip_unless_1n1d()
class TestBitwiseOrInplaceModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_or_inplace(test_case):
        return _test_bitwise_inplace_op(test_case, torch.bitwise_or,flow.bitwise_or_)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_or_inplace(test_case):
        return _test_scalar_bitwise_inplace(test_case, torch.bitwise_or,flow.bitwise_or_)


@flow.unittest.skip_unless_1n1d()
class TestBitwiseXorModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_xor(test_case):
        return _test_bitwise_op(test_case, torch.bitwise_xor)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_xor(test_case):
        return _test_scalar_bitwise(test_case, torch.bitwise_xor,)

@flow.unittest.skip_unless_1n1d()
class TestBitwiseXorInplaceModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_xor_inplace(test_case):
        return _test_bitwise_inplace_op(test_case, torch.bitwise_xor,flow.bitwise_xor_)

    @autotest(n=10, auto_backward=False)
    def test_scalar_bitwise_xor_inplace(test_case):
        return _test_scalar_bitwise_inplace(test_case, torch.bitwise_xor,flow.bitwise_xor_)

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


@flow.unittest.skip_unless_1n1d()
class TestBitwiseNotInplaceModule(flow.unittest.TestCase):
    @autotest(n=10, auto_backward=False)
    def test_bitwise_not_inplace(test_case):
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
        result = torch.bitwise_not(x)
        
        x_flow = x.oneflow.clone()
        x_flow.bitwise_not_()
        test_case.assertTrue(
            np.allclose(
                x_flow.numpy(),
                result.pytorch.cpu().numpy()
            )
        )
    

if __name__ == "__main__":
    unittest.main()

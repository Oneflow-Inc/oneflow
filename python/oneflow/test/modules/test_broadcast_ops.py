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

binary_ops = [
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.min,
    torch.minimum,
    torch.max,
    torch.maximum,
    torch.fmod,
    torch.pow,
    torch.eq,
    torch.ne,
    torch.gt,
    torch.ge,
    torch.lt,
    torch.le,
    torch.logical_and,
    torch.logical_or,
    torch.logical_xor,
]


@flow.unittest.skip_unless_1n1d()
class TestBroadcastOps(flow.unittest.TestCase):
    @autotest(n=5, auto_backward=False)
    def test_broadcast_elementwise(test_case):
        op_idx = random(low=0, high=len(binary_ops)).to(int).value()
        op = binary_ops[op_idx]
        device = random_device()
        x = random_tensor(ndim=4, dim0=2, dim1=2, dim2=3, dim3=4).to(device)
        y = random_tensor(ndim=4, dim0=1, dim1=2, dim2=3, dim3=1).to(device)
        out = op(x, y)
        return out

    @autotest(n=5, auto_backward=False)
    def test_broadcast_matrix_row(test_case):
        op_idx = random(low=0, high=len(binary_ops)).to(int).value()
        op = binary_ops[op_idx]
        device = random_device()
        x = random_tensor(ndim=3, dim0=2, dim1=2, dim2=3).to(device)
        y = random_tensor(ndim=2, dim0=2, dim1=3).to(device)
        out = op(x, y)
        return out

    @autotest(n=5, auto_backward=False)
    def test_broadcast_matrix_col(test_case):
        op_idx = random(low=0, high=len(binary_ops)).to(int).value()
        op = binary_ops[op_idx]
        device = random_device()
        x = random_tensor(ndim=3, dim0=2, dim1=2, dim2=3).to(device)
        y = random_tensor(ndim=3, dim0=2, dim1=2, dim2=1).to(device)
        out = op(x, y)
        return out

    @autotest(n=30, auto_backward=False)
    def test_broadcast_scalar(test_case):
        op_idx = random(low=0, high=len(binary_ops)).to(int).value()
        op = binary_ops[op_idx]
        device = random_device()
        x = random_tensor(ndim=3, dim0=2, dim1=2, dim2=3).to(device)
        out = op(x, 1)
        return out

    @profile(torch.add)
    def profile_broadcast_matrix_row(test_case):
        input0 = torch.ones(256, 1024)
        input1 = torch.ones(1024)
        torch.add(input0, input1)

    @profile(torch.add)
    def profile_broadcast_matrix_col(test_case):
        input0 = torch.ones(1024, 256)
        input1 = torch.ones(1024, 1)
        torch.add(input0, input1)

    @profile(torch.add)
    def profile_broadcast_elementwise(test_case):
        input0 = torch.ones(256, 1024)
        input1 = torch.ones(256, 1024)
        torch.add(input0, input1)

    @profile(torch.add)
    def profile_broadcast_scalar(test_case):
        input0 = torch.ones(256, 1024)
        torch.add(input0, 1)

    @profile(torch.add)
    def profile_broadcast_general(test_case):
        input0 = torch.ones(2, 64, 8, 16, 16, 4)
        input1 = torch.ones(64, 8, 1, 16, 1)
        torch.add(input0, input1)


if __name__ == "__main__":
    unittest.main()

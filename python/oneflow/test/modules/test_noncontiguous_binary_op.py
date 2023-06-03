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
from oneflow.test_utils.test_util import GenArgList


def _test_op(test_case, x, y, inplace):
    ref1 = x + y
    out1 = flow._C.noncontiguous_binary_op(x, y, op="add", inplace=inplace)
    test_case.assertTrue(np.allclose(ref1.numpy(), out1.numpy(), rtol=1e-5, atol=1e-5))

    ref2 = x - y
    out2 = flow._C.noncontiguous_binary_op(x, y, op="sub", inplace=inplace)
    test_case.assertTrue(np.allclose(ref2.numpy(), out2.numpy(), rtol=1e-5, atol=1e-5))

    ref3 = x * y
    out3 = flow._C.noncontiguous_binary_op(x, y, op="mul", inplace=inplace)
    test_case.assertTrue(np.allclose(ref3.numpy(), out3.numpy(), rtol=1e-5, atol=1e-5))

    y = y.abs() + 1e-3  # incase zero
    ref4 = x / y
    out4 = flow._C.noncontiguous_binary_op(x, y, op="div", inplace=inplace)
    print(np.abs(ref4 - out4).max())
    test_case.assertTrue(np.allclose(ref4.numpy(), out4.numpy(), rtol=1e-3, atol=1e-3))


def _test_noncontiguous_binary_op(test_case, dtype, pack_size, ndims, inplace):
    shape = []
    for _ in range(ndims - 1):
        if np.random.uniform(-1, 1) > 0:
            shape.append(1 << np.random.randint(4, 7))
        else:
            shape.append(np.random.randint(20, 100))
    shape.append(1 << np.random.randint(3, 7) + pack_size)
    # case 1
    x = flow.randn(*shape, requires_grad=True).cuda().to(dtype)
    y = flow.randn(*shape, requires_grad=True).cuda().to(dtype)
    d1, d2 = np.random.choice(ndims, 2, replace=False)
    x1 = x.transpose(d1, d2)
    y1 = y.transpose(d1, d2)
    _test_op(test_case, x1, y1, inplace)

    # case 2
    y2 = flow.randn(*shape, requires_grad=True).cuda().to(dtype)
    shape[d1], shape[d2] = shape[d2], shape[d1]
    x = flow.randn(*shape, requires_grad=True).cuda().to(dtype)
    x2 = x.transpose(d1, d2)
    _test_op(test_case, x2, y2, inplace)


@unittest.skipIf(True, "skip test for noncontiguous_binary_op.")
@flow.unittest.skip_unless_1n1d()
class TestNonContiguousBinaryOp(flow.unittest.TestCase):
    def test_noncontiguous_binary_op(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fn"] = [_test_noncontiguous_binary_op]
        arg_dict["dtype"] = [flow.float16, flow.float32]
        arg_dict["pack_size"] = [1, 2, 4]
        arg_dict["ndims"] = [2, 3, 4]
        arg_dict["inplace"] = [True, False]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

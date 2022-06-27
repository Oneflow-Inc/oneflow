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
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import (
    GenArgList,
    type_name_to_flow_type,
    type_name_to_np_type,
)


@flow.unittest.skip_unless_1n1d()
class TestSinh(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_sinh_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.sinh(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestSin(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_sin_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.sin()
        return y


@flow.unittest.skip_unless_1n1d()
class TestInplaceSin(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_inplace_sin_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x + 1  # transform to non-leaf tensor
        y.sin_()
        return y


def _test_cos(test_case, shape, device):
    input = flow.tensor(
        np.random.randn(*shape), dtype=flow.float32, device=flow.device(device)
    )
    of_out = flow.cos(input)
    np_out = np.cos(input.numpy())
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_cos_backward(test_case, shape, device):
    x = flow.tensor(
        np.random.randn(*shape),
        dtype=flow.float32,
        device=flow.device(device),
        requires_grad=True,
    )
    y = flow.cos(x)
    z = y.sum()
    z.backward()
    np_grad = -np.sin(x.numpy())
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestCos(flow.unittest.TestCase):
    def test_cos(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_cos, _test_cos_backward]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


@flow.unittest.skip_unless_1n1d()
class TestLogModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_log_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return torch.log(x)


@flow.unittest.skip_unless_1n1d()
class TestSqrt(flow.unittest.TestCase):
    @autotest(n=5)
    def test_sqrt_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        z = torch.sqrt(x)
        return z

    @autotest(n=5)
    def test_sqrt_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        z = x.sqrt()
        return z


@flow.unittest.skip_unless_1n1d()
class TestExp(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_exp_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.exp(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestRsqrt(flow.unittest.TestCase):
    @autotest(n=5)
    def test_rsqrt_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        z = torch.rsqrt(x)
        return z


@flow.unittest.skip_unless_1n1d()
class TestSquare(flow.unittest.TestCase):
    @autotest(n=5)
    def test_square_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        z = torch.square(x)
        return z

    @autotest(n=5)
    def test_square_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        z = x.square()
        return z


@flow.unittest.skip_unless_1n1d()
class TestPow(flow.unittest.TestCase):
    @autotest(n=5)
    def test_pow_float_scalar_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = random().to(float)
        return torch.pow(x, y)

    def test_pow_int_scalar_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = random().to(int)
        return torch.pow(x, y)

    @autotest(n=10)
    def test_reverse_pow_int_scalar_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = random().to(int)
        return torch.pow(y, x)

    @autotest(n=10)
    def test_symbolic_reverse_pow_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = random().to(int)
        return y ** x

    @autotest(n=5)
    def test_pow_elementwise_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=2).to(device)
        y = random_tensor(ndim=2, dim1=2).to(device)
        return torch.pow(x, y)

    @autotest(n=5)
    def test_pow_broadcast_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=2).to(device)
        y = random_tensor(ndim=2, dim1=1).to(device)
        return torch.pow(x, y)

    @autotest(n=5)
    def test_pow_broadcast_with_random_data_reverse(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=1).to(device)
        y = random_tensor(ndim=2, dim1=2).to(device)
        return torch.pow(x, y)


@flow.unittest.skip_unless_1n1d()
class TestAsin(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_asin_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-0.5, high=0.5).to(device)
        y = torch.asin(x)
        return y

    @autotest(n=5)
    def test_flow_arcsin_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-0.5, high=0.5).to(device)
        y = torch.arcsin(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestAsinh(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_asinh_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.asinh(x)
        return y

    @autotest(n=5)
    def test_flow_arcsinh_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.arcsinh(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestTan(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_tan_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.tan(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestAtan(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_atan_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.atan(x)
        return y

    @autotest(n=5)
    def test_flow_arctan_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.arctan(x)
        return y

    @autotest(n=5)
    def test_flow_atan2_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=3).to(device)
        y = random_tensor(ndim=2, dim1=3).to(device)
        z = torch.atan2(x, y)
        return z

    @autotest(n=5)
    def test_flow_atanh_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-0.5, high=0.5).to(device)
        y = torch.atanh(x)
        return y

    @autotest(n=5)
    def test_flow_arctanh_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-0.5, high=0.5).to(device)
        y = torch.arctanh(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestTopk(flow.unittest.TestCase):
    @autotest(auto_backward=False)
    def test_flow_topk_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dim1=8, dim2=9, dim3=10).to(device)
        y = torch.topk(
            x,
            random(low=1, high=8).to(int),
            dim=random(low=1, high=4).to(int),
            largest=random_bool(),
            sorted=constant(True),
        )
        return y[0], y[1]


@flow.unittest.skip_unless_1n1d()
class TestPow(flow.unittest.TestCase):
    @autotest(n=5)
    def test_pow_scalar_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = random().to(float)
        return torch.pow(x, y)

    @autotest(n=5)
    def test_pow_elementwise_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=2).to(device)
        y = random_tensor(ndim=2, dim1=2).to(device)
        return torch.pow(x, y)

    @unittest.skip("not support for broadcast currently")
    @autotest(n=5)
    def test_pow_broadcast_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=2).to(device)
        y = random_tensor(ndim=2, dim1=1).to(device)
        return torch.pow(x, y)


@flow.unittest.skip_unless_1n1d()
class TestArccos(flow.unittest.TestCase):
    @autotest(n=5)
    def test_arccos_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-1, high=1).to(device)
        y = torch.arccos(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestAcos(flow.unittest.TestCase):
    @autotest(n=5)
    def test_acos_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-1, high=1).to(device)
        y = torch.acos(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestArccosh(flow.unittest.TestCase):
    @autotest(n=5)
    def test_arccosh_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=2, high=3).to(device)
        y = torch.arccosh(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestAcosh(flow.unittest.TestCase):
    @autotest(n=5)
    def test_acosh_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=2, high=3).to(device)
        y = torch.acosh(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestAtan2(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_atan2_with_random_data(test_case):
        device = random_device()
        x1 = random_tensor(ndim=1, dim0=1).to(device)
        x2 = random_tensor(ndim=1, dim0=1).to(device)
        y = torch.atan2(x1, x2)
        return y


@flow.unittest.skip_unless_1n1d()
class TestMinimum(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_elementwise_minimum_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        x = random_tensor(ndim=2, dim0=k1, dim1=k2)
        y = random_tensor(ndim=2, dim0=k1, dim1=k2)
        return torch.minimum(x, y)

    @autotest(n=5)
    def test_flow_broadcast_minimum_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        k3 = random(2, 6)
        x = random_tensor(ndim=3, dim0=k1, dim1=1, dim2=1)
        y = random_tensor(ndim=3, dim0=1, dim1=k2, dim2=k3)
        return torch.minimum(x, y)


@flow.unittest.skip_unless_1n1d()
class TestMaximum(flow.unittest.TestCase):
    @autotest(n=5)
    def test_flow_elementwise_mximum_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        x = random_tensor(ndim=2, dim0=k1, dim1=k2)
        y = random_tensor(ndim=2, dim0=k1, dim1=k2)
        return torch.maximum(x, y)

    @autotest(n=5)
    def test_flow_broadcast_maximum_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        k3 = random(2, 6)
        x = random_tensor(ndim=3, dim0=k1, dim1=1, dim2=1)
        y = random_tensor(ndim=3, dim0=1, dim1=k2, dim2=k3)
        return torch.maximum(x, y)


@flow.unittest.skip_unless_1n1d()
class TestFloorDiv(flow.unittest.TestCase):
    @autotest(auto_backward=False)
    def test_elementwise_floordiv_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dim0=2, dim1=4, dim2=8, dim3=3).to(device)
        y = random_tensor(ndim=4, dim0=2, dim1=4, dim2=8, dim3=3).to(device)

        return torch.floor_divide(x, y)

    @autotest(auto_backward=False)
    def test_tensor_floordiv_scalar_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dim0=2, dim1=4, dim2=8, dim3=3).to(device)
        y = random().to(int)
        return torch.floor_divide(x, y)


if __name__ == "__main__":
    unittest.main()

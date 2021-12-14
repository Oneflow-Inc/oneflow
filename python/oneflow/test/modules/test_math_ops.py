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

from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


@flow.unittest.skip_unless_1n1d()
class TestSinh(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_sinh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.sinh(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_sinh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.sinh()
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_sinh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.sinh_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_sinh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.sinh_()
        test_case.assertTrue(id_x, id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestSin(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_sin_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.sin(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_sin_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.sin()
        return y


@flow.unittest.skip_unless_1n1d()
class TestInplaceSin(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_tensor_inplace_sin_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.sin()
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_sin_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1  # transform to non-leaf tensor
        id_x = id(x)
        torch.sin_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_sin_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.sin_()
        test_case.assertTrue(id_x == id(x))
        return x


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

    @autotest(check_graph=False)
    def test_flow_inplace_cos_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.cos_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_cos_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.cos_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestLogModule(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_log_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.log(x)
        return y

    @autotest(check_graph=False)
    def test_log_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        return torch.log(x)

    @autotest(check_graph=False)
    def test_tensor_log_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.log()
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_log_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.log_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_log_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.log_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestSqrt(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_sqrt_flow_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        z = torch.sqrt(x)
        return z

    @autotest(check_graph=False)
    def test_sqrt_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        z = x.sqrt()
        return z

    @autotest(check_graph=False)
    def test_flow_inplace_sqrt_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.sqrt_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_sqrt_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.sqrt_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestExp(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_exp_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.exp(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_exp_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.exp(x)
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_exp_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.exp_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_exp_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.exp_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestRsqrt(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_rsqrt_flow_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        z = torch.rsqrt(x)
        return z

    @autotest(check_graph=False)
    def test_rsqrt_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        z = x.rsqrt()
        return z

    @autotest(check_graph=False)
    def test_flow_inplace_rsqrt_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.rsqrt_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_rsqrt_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.rsqrt_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestSquare(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_square_flow_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        z = torch.square(x)
        return z

    @autotest(check_graph=False)
    def test_square_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        z = x.square()
        return z

    @autotest(check_graph=False)
    def test_flow_inplace_square_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.square_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_square_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.square_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestPow(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_pow_scalar_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = random().to(float)
        return torch.pow(x, y)

    @autotest(check_graph=False)
    def test_pow_elementwise_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2, dim1=2).to(device)
        y = random_pytorch_tensor(ndim=2, dim1=2).to(device)
        return torch.pow(x, y)

    @autotest(check_graph=False)
    def test_pow_broadcast_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2, dim1=2).to(device)
        y = random_pytorch_tensor(ndim=2, dim1=1).to(device)
        return torch.pow(x, y)

    @autotest(check_graph=False)
    def test_pow_broadcast_with_random_data_reverse(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2, dim1=1).to(device)
        y = random_pytorch_tensor(ndim=2, dim1=2).to(device)
        return torch.pow(x, y)


@flow.unittest.skip_unless_1n1d()
class TestAsin(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_asin_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = torch.asin(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_asin_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = x.asin()
        return y

    @autotest(check_graph=False)
    def test_flow_arcsin_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = torch.arcsin(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_arcsin_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = x.arcsin()
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_asin_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.asin_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_asin_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.asin_()
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_flow_inplace_arcsin_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.arcsin_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_arcsin_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.arcsin_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestAsinh(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_asinh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.asinh(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_asinh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.asinh()
        return y

    @autotest(check_graph=False)
    def test_flow_arcsinh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.arcsinh(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_arcsinh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.arcsinh()
        return y

    @autotest(auto_backward=False, check_graph=False)
    def test_flow_inplace_asinh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.asinh_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(auto_backward=False, check_graph=False)
    def test_tensor_inplace_asinh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.asinh_()
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(auto_backward=False, check_graph=False)
    def test_flow_inplace_arcsinh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.arcsinh_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(auto_backward=False, check_graph=False)
    def test_tensor_inplace_arcsinh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.arcsinh_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestTan(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_tan_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.tan(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_tan_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.tan()
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_tan_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.tan_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_tan_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.tan_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestTanh(flow.unittest.TestCase):
    @autotest
    def test_flow_tanh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.tanh(x)
        return y

    @autotest
    def test_tensor_tanh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.tanh(x)
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_tanh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.tanh_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_tanh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.tanh_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestAtan(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_atan_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.atan(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_atan_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.atan()
        return y

    @autotest(check_graph=False)
    def test_flow_arctan_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = torch.arctan(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_arctan_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.arctan()
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_atan_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.atan_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_atan_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.atan_()
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_flow_inplace_arctan_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.arctan_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_arctan_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.arctan_()
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_flow_atan2_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2, dim1=3).to(device)
        y = random_pytorch_tensor(ndim=2, dim1=3).to(device)
        z = torch.atan2(x, y)
        return z

    @autotest(check_graph=False)
    def test_flow_atanh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = torch.atanh(x)
        return y

    @autotest(check_graph=False)
    def test_tensor_atanh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = x.atanh()
        return y

    @autotest(check_graph=False)
    def test_flow_arctanh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = torch.arctanh(x)
        return y

    @autotest(auto_backward=False, check_graph=False)
    def test_tensor_arctanh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = torch.arctanh(x)
        return y

    @autotest(auto_backward=False, check_graph=False)
    def test_flow_inplace_atanh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.atanh_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(auto_backward=False, check_graph=False)
    def test_tensor_inplace_atanh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.atanh_()
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(auto_backward=False, check_graph=False)
    def test_flow_inplace_arctanh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.arctanh_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(auto_backward=False, check_graph=False)
    def test_tensor_inplace_arctanh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.arctanh_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestTopk(flow.unittest.TestCase):
    @autotest(auto_backward=False, check_graph=False)
    def test_flow_topk_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim1=8, dim2=9, dim3=10).to(device)
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
    @autotest(check_graph=False)
    def test_pow_scalar_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = random().to(float)
        return torch.pow(x, y)

    @autotest(check_graph=False)
    def test_pow_elementwise_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2, dim1=2).to(device)
        y = random_pytorch_tensor(ndim=2, dim1=2).to(device)
        return torch.pow(x, y)

    @unittest.skip("not support for broadcast currently")
    @autotest()
    def test_pow_broadcast_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2, dim1=2).to(device)
        y = random_pytorch_tensor(ndim=2, dim1=1).to(device)
        return torch.pow(x, y)


@flow.unittest.skip_unless_1n1d()
class TestArccos(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_arccos_flow_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=2, high=3).to(device)
        y = torch.arccos(x)
        return y

    @autotest(check_graph=False)
    def test_arccos_tensor_with_random_data(test_case, check_graph=False):
        device = random_device()
        x = random_pytorch_tensor(low=2, high=3).to(device)
        y = x.arccos()
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_arccos_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.arccos_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_arccos_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.arccos_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestAcos(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_acos_flow_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=2, high=3).to(device)
        y = torch.acos(x)
        return y

    @autotest(check_graph=False)
    def test_acos_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=2, high=3).to(device)
        y = x.acos()
        return y

    @autotest(check_graph=False)
    def test_flow_inplace_acos_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.acos_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(check_graph=False)
    def test_tensor_inplace_acos_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.acos_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestArccosh(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_arccosh_flow_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=2, high=3).to(device)
        y = torch.arccosh(x)
        return y

    @autotest(check_graph=False)
    def test_arccosh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=2, high=3).to(device)
        y = x.arccosh()
        return y

    @autotest(auto_backward=False, check_graph=False)
    def test_flow_inplace_arccosh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.arccosh_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(auto_backward=False, check_graph=False)
    def test_tensor_inplace_arccosh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.arccosh_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestAcosh(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_acosh_flow_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=2, high=3).to(device)
        y = torch.acosh(x)
        return y

    @autotest(check_graph=False)
    def test_acosh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=2, high=3).to(device)
        y = x.acosh()
        return y

    @autotest(auto_backward=False, check_graph=False)
    def test_flow_inplace_acosh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        torch.acosh_(x)
        test_case.assertTrue(id_x == id(x))
        return x

    @autotest(auto_backward=False, check_graph=False)
    def test_tensor_inplace_acosh_with_random_data(test_case):
        device = random_device()
        x_0 = random_pytorch_tensor().to(device)
        x = x_0 + 1
        id_x = id(x)
        x.acosh_()
        test_case.assertTrue(id_x == id(x))
        return x


@flow.unittest.skip_unless_1n1d()
class TestAtan2(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_atan2_with_random_data(test_case):
        device = random_device()
        x1 = random_pytorch_tensor(ndim=1, dim0=1).to(device)
        x2 = random_pytorch_tensor(ndim=1, dim0=1).to(device)
        y = torch.atan2(x1, x2)
        return y


@flow.unittest.skip_unless_1n1d()
class TestMinimum(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_elementwise_minimum_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        x = random_pytorch_tensor(ndim=2, dim0=k1, dim1=k2)
        y = random_pytorch_tensor(ndim=2, dim0=k1, dim1=k2)
        return torch.minimum(x, y)

    @autotest(check_graph=False)
    def test_flow_broadcast_minimum_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        k3 = random(2, 6)
        x = random_pytorch_tensor(ndim=3, dim0=k1, dim1=1, dim2=1)
        y = random_pytorch_tensor(ndim=3, dim0=1, dim1=k2, dim2=k3)
        return torch.minimum(x, y)


class TestMaximum(flow.unittest.TestCase):
    @autotest(check_graph=False)
    def test_flow_elementwise_mximum_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        x = random_pytorch_tensor(ndim=2, dim0=k1, dim1=k2)
        y = random_pytorch_tensor(ndim=2, dim0=k1, dim1=k2)
        return torch.maximum(x, y)

    @autotest(check_graph=False)
    def test_flow_broadcast_maximum_with_random_data(test_case):
        k1 = random(2, 6)
        k2 = random(2, 6)
        k3 = random(2, 6)
        x = random_pytorch_tensor(ndim=3, dim0=k1, dim1=1, dim2=1)
        y = random_pytorch_tensor(ndim=3, dim0=1, dim1=k2, dim2=k3)
        return torch.maximum(x, y)


@flow.unittest.skip_unless_1n1d()
class TestFloorDiv(flow.unittest.TestCase):
    @autotest(auto_backward=False, check_graph=False)
    def test_elementwise_floordiv_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim0=2, dim1=4, dim2=8, dim3=3).to(device)
        y = random_pytorch_tensor(ndim=4, dim0=2, dim1=4, dim2=8, dim3=3).to(device)

        return torch.floor_divide(x, y)

    @autotest(auto_backward=False, check_graph=False)
    def test_tensor_floordiv_scalar_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4, dim0=2, dim1=4, dim2=8, dim3=3).to(device)
        y = random().to(int)
        return torch.floor_divide(x, y)


if __name__ == "__main__":
    unittest.main()

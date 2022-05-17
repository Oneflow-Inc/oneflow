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
from scipy import special
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestReLUModule(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_relu_module_with_random_data(test_case):
        m = torch.nn.ReLU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_relu_module_with_0dim_data(test_case):
        m = torch.nn.ReLU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_relu_module_with_0_size_data(test_case):
        m = torch.nn.ReLU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestReLU6Module(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_relu6_module_with_random_data(test_case):
        m = torch.nn.ReLU6()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_relu6_module_with_0dim_data(test_case):
        m = torch.nn.ReLU6()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_relu6_module_with_0_size_data(test_case):
        m = torch.nn.ReLU6()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestTanh(flow.unittest.TestCase):
    @autotest()
    def test_tanh_module_with_random_data(test_case):
        m = torch.nn.Tanh()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_tanh_module_with_0dim_data(test_case):
        m = torch.nn.Tanh()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_tanh_module_with_0_size_data(test_case):
        m = torch.nn.Tanh()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y

    @autotest(check_graph=True)
    def test_flow_tanh_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.tanh(x)
        return y

    @autotest(check_graph=True)
    def test_flow_tanh_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.tanh(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_tanh_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 3, 0, 3).to(device)
        y = torch.tanh(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestELUModule(flow.unittest.TestCase):
    @autotest()
    def test_elu_module_with_random_data(test_case):
        m = torch.nn.ELU(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_elu_module_with_0dim_data(test_case):
        m = torch.nn.ELU(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_elu_module_with_0_size_data(test_case):
        m = torch.nn.ELU(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestCELUModule(flow.unittest.TestCase):
    @autotest()
    def test_celu_module_with_random_data(test_case):
        m = torch.nn.CELU(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_celu_module_with_0dim_data(test_case):
        m = torch.nn.CELU(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_celu_module_with_0_size_data(test_case):
        m = torch.nn.CELU(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y

    @autotest(n=10)
    def test_inplace_celu_module(test_case):
        m = torch.nn.CELU(alpha=random() | nothing(), inplace=True)
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = x + 0.001
        m(y)
        return y


@flow.unittest.skip_unless_1n1d()
class TestGelu(flow.unittest.TestCase):
    @autotest()
    def test_gelu_module_with_random_data(test_case):
        m = torch.nn.GELU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_gelu_module_with_0dim_data(test_case):
        m = torch.nn.GELU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestSigmoidModule(flow.unittest.TestCase):
    @autotest()
    def test_sigmoid_module_with_random_data(test_case):
        m = torch.nn.Sigmoid()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_sigmoid_module_with_0dim_data(test_case):
        m = torch.nn.Sigmoid()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y

    @autotest(check_graph=True)
    def test_sigmoid_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.sigmoid(x)
        return y

    @autotest(check_graph=True)
    def test_sigmoid_flow_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.sigmoid(x)
        return y

    @autotest(check_graph=True)
    def test_sigmoid_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.sigmoid()
        return y

    @autotest(check_graph=True)
    def test_sigmoid_tensor_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = x.sigmoid()
        return y


@flow.unittest.skip_unless_1n1d()
class TestHardsigmoidModule(flow.unittest.TestCase):
    def test_hardsigmoid_inplace(test_case):
        def np_hardsigmoid(input):
            input_shape = input.shape
            input = input.flatten()
            elem_cnt = input.size
            _zero = np.zeros_like(input)
            for i in range(elem_cnt):
                if input[i] >= 3:
                    _zero[i] = 1
                elif input[i] <= -3:
                    _zero[i] = 0
                else:
                    _zero[i] = input[i] / 6 + 0.5
            np_hsigmoid_out = np.reshape(_zero, newshape=input_shape)
            return np.array(np_hsigmoid_out)

        def test_hardsigmoid_inplace_impl(test_case, shape, device):
            x = flow.tensor(
                np.random.randn(*shape),
                dtype=flow.float32,
                device=flow.device(device),
                requires_grad=True,
            )
            x_inplace = x + 1
            np_out = np_hardsigmoid(x_inplace.numpy())

            id_old = id(x_inplace)
            y_inplace = flow.nn.functional.hardsigmoid(x_inplace, inplace=True)

            test_case.assertEqual(id_old, id(y_inplace))
            test_case.assertTrue(np.allclose(y_inplace.numpy(), np_out, 1e-5, 1e-5))

        arg_dict = OrderedDict()
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            test_hardsigmoid_inplace_impl(test_case, *arg)

    @autotest()
    def test_hardsigmoid_module_with_random_data(test_case):
        m = torch.nn.Hardsigmoid()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_hardsigmoid_module_with_0dim_data(test_case):
        m = torch.nn.Hardsigmoid()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y

    @autotest(check_graph=True)
    def test_functional_hardsigmoid_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = torch.nn.functional.hardsigmoid(x, random_bool())
        return y

    @autotest(check_graph=True)
    def test_functional_hardsigmoid_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.nn.functional.hardsigmoid(x, random_bool())
        return y


def do_test_softmax(batch_size: int, log_softmax: bool = False):
    num_dims = random(low=1, high=5).to(int)
    m = torch.nn.Softmax(dim=random(low=0, high=num_dims).to(int) | nothing())
    if log_softmax:
        m = torch.nn.LogSoftmax(dim=random(low=0, high=num_dims).to(int) | nothing())
    m.train(random())
    device = random_device()
    m.to(device)
    x = (
        random_tensor(ndim=num_dims).to(device)
        if batch_size < 0
        else random_tensor(ndim=num_dims, dim0=batch_size).to(device)
    )
    y = m(x)
    return y


@flow.unittest.skip_unless_1n1d()
class TestSoftmax(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_softmax_module_with_random_data(test_case):
        return do_test_softmax(batch_size=-1, log_softmax=False)

    @autotest(check_graph=True)
    def test_softmax_module_with_batch_size_equal_1024(test_case):
        return do_test_softmax(batch_size=1024, log_softmax=False)

    @autotest(n=5, check_graph=True)
    def test_softmax_module_with_batch_size_equal_5120(test_case):
        return do_test_softmax(batch_size=5120, log_softmax=False)

    @autotest(n=2, check_graph=True)
    def test_softmax_module_with_batch_size_equal_10240(test_case):
        return do_test_softmax(batch_size=10240, log_softmax=False)


@flow.unittest.skip_unless_1n1d()
class TestLogSoftmaxModule(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_logsoftmax_module_with_random_data(test_case):
        return do_test_softmax(batch_size=-1, log_softmax=True)

    @autotest()
    def test_softmax_module_with_batch_size_equal_1024(test_case):
        return do_test_softmax(batch_size=1024, log_softmax=True)

    @autotest(n=5, check_graph=True)
    def test_softmax_module_with_batch_size_equal_5120(test_case):
        return do_test_softmax(batch_size=5120, log_softmax=True)

    @autotest(n=2, check_graph=True)
    def test_softmax_module_with_batch_size_equal_10240(test_case):
        return do_test_softmax(batch_size=10240, log_softmax=True)


@flow.unittest.skip_unless_1n1d()
class TestLogSigmoidModule(flow.unittest.TestCase):
    @autotest()
    def test_logsigmoid_module_with_random_data(test_case):
        m = torch.nn.LogSigmoid()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_logsigmoid_module_with_0dim_data(test_case):
        m = torch.nn.LogSigmoid()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y


def numpy_softplus(x, beta, threshold):
    return np.where(
        x * beta > threshold, x, 1.0 / beta * np.log(1.0 + np.exp(beta * x))
    )


def _test_softplus(test_case, device):
    m = flow.nn.Softplus()
    arr = np.random.randn(2, 3, 4, 5)
    np_out = numpy_softplus(arr, 1.0, 20)
    x = flow.tensor(arr, device=flow.device(device))
    of_out = m(x)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_softplus_beta(test_case, device):
    m = flow.nn.Softplus(beta=1.11)
    arr = np.random.randn(2, 3, 4, 5)
    np_out = numpy_softplus(arr, 1.11, 20)
    x = flow.tensor(arr, device=flow.device(device))
    of_out = m(x)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_softplus_threshold(test_case, device):
    m = flow.nn.Softplus(beta=1.11, threshold=1.55)
    arr = np.random.randn(2, 3, 4, 5)
    np_out = np.where(
        arr * 1.11 > 1.55, arr, 1.0 / 1.11 * np.log(1.0 + np.exp(1.11 * arr))
    )
    np_out = numpy_softplus(arr, 1.11, 1.55)
    x = flow.tensor(arr, device=flow.device(device))
    of_out = m(x)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))


def _test_softplus_backward(test_case, device):
    m = flow.nn.Softplus()
    arr = np.array([1.0, 2.0, 21.0, 20.0, 4.0])
    x = flow.tensor(arr, device=flow.device(device), requires_grad=True)
    of_out = m(x)
    of_out = of_out.sum()
    of_out.backward()
    np_grad = [0.7310585786300049, 0.8807970779778824, 1.0, 1.0, 0.9820137900379085]
    test_case.assertTrue(np.allclose(x.grad.numpy(), np_grad, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestSoftplusModule(flow.unittest.TestCase):
    def test_softplus(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_softplus,
            _test_softplus_beta,
            _test_softplus_threshold,
            _test_softplus_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @unittest.skip("pytorch softplus backward has bug")
    @autotest()
    def test_softplus_module_with_random_data(test_case):
        m = torch.nn.Softplus(beta=random() | nothing(), threshold=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestHardswishModule(flow.unittest.TestCase):
    @autotest()
    def test_hardswish_module_with_random_data(test_case):
        m = torch.nn.Hardswish()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    def test_hardswish_module_with_0dim_data(test_case):
        m = torch.nn.Hardswish()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestHardtanhModule(flow.unittest.TestCase):
    @autotest()
    def test_hardtanh_module_with_random_data(test_case):
        m = torch.nn.Hardtanh(
            min_val=random().to(float) | nothing(),
            max_val=random().to(float) | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=4).to(device)
        y = m(x)
        return y

    @autotest()
    def test_hardtanh_module_with_0dim_data(test_case):
        m = torch.nn.Hardtanh(
            min_val=random().to(float) | nothing(),
            max_val=random().to(float) | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestLeakyReLUModule(flow.unittest.TestCase):
    @autotest()
    def test_leakyrelu_module_with_random_data(test_case):
        m = torch.nn.LeakyReLU(negative_slope=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest(n=5)
    def test_leakyrelu_module_with_inplace_arg(test_case):
        m = torch.nn.LeakyReLU(
            negative_slope=random() | nothing(), inplace=random().to(bool) | nothing()
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_leakyrelu_module_with_half_random_data(test_case):
        m = torch.nn.LeakyReLU(negative_slope=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        x = x.to(torch.float16)
        y = m(x)
        return y

    @autotest()
    def test_leakyrelu_module_with_0dim_data(test_case):
        m = torch.nn.LeakyReLU(negative_slope=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestMishModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_mish_module_with_random_data(test_case):
        m = torch.nn.Mish()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest(n=5)
    def test_mish_module_with_0dim_data(test_case):
        m = torch.nn.Mish()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestSiluModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_silu_module_with_random_data(test_case):
        m = torch.nn.SiLU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest(n=5)
    def test_silu_module_with_0dim_data(test_case):
        m = torch.nn.SiLU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestSeluModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_selu_module_with_random_data(test_case):
        m = torch.nn.SELU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest(n=5)
    def test_selu_module_with_0dim_data(test_case):
        m = torch.nn.SELU()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y


@unittest.skip("still have error in ci test")
class TestSoftsignModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_softsign_module_with_random_data(test_case):
        m = torch.nn.Softsign()
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestReluFunction(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_relu_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=3).to(device)
        y = torch.relu(x)
        return y

    @autotest(check_graph=True)
    def test_flow_relu_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.relu(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestRelu6Function(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_nn_functional_relu6_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=3).to(device)
        y = torch.nn.functional.relu6(x)
        return y

    @autotest(check_graph=True)
    def test_flow_nn_functional_relu6_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.nn.functional.relu6(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestLogSigmoidFunction(flow.unittest.TestCase):
    @autotest(check_graph=True)
    def test_flow_nn_functional_logsigmoid_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=3).to(device)
        y = torch.nn.functional.logsigmoid(x)
        return y

    @autotest(check_graph=True)
    def test_flow_nn_functional_logsigmoid_with_0dim_data(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        y = torch.nn.functional.logsigmoid(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestHardshrinkModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_hardshrink_module_with_random_data(test_case):
        m = torch.nn.Hardshrink(lambd=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest(n=5)
    def test_hardshrink_module_with_0dim_data(test_case):
        m = torch.nn.Hardshrink(lambd=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_hardshrink_module_with_0_size_data(test_case):
        m = torch.nn.Hardshrink(lambd=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestSoftshrinkModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_softshrink_module_with_random_data(test_case):
        m = torch.nn.Softshrink(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest(n=5)
    def test_softshrink_module_with_0dim_data(test_case):
        m = torch.nn.Softshrink(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_softshrink_module_with_0_size_data(test_case):
        m = torch.nn.Softshrink(alpha=random() | nothing())
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y


@flow.unittest.skip_unless_1n1d()
class TestThresholdModule(flow.unittest.TestCase):
    @autotest(n=5)
    def test_threshold_module_with_random_data(test_case):
        m = torch.nn.Threshold(
            threshold=random() | nothing(), value=random() | nothing()
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor().to(device)
        y = m(x)
        return y

    @autotest(n=5)
    def test_threshold_module_with_0dim_data(test_case):
        m = torch.nn.Threshold(
            threshold=random() | nothing(), value=random() | nothing()
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=0).to(device)
        y = m(x)
        return y

    @autotest(auto_backward=False, check_graph=True)
    def test_threshold_module_with_0_size_data(test_case):
        m = torch.nn.Threshold(
            threshold=random() | nothing(), value=random() | nothing()
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(4, 2, 3, 0, 3).to(device)
        y = m(x)
        return y


if __name__ == "__main__":
    unittest.main()

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
from test_util import GenArgList
from collections import OrderedDict
import numpy as np
import oneflow.experimental as flow


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestReLUModule(flow.unittest.TestCase):
    def test_relu(test_case):
        m = flow.nn.ReLU()
        arr = np.random.randn(2, 3, 4, 5)

        np_out = np.maximum(0, arr)
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestReLU6Module(flow.unittest.TestCase):
    def test_relu6(test_case):
        m = flow.nn.ReLU6()
        arr = np.random.randn(2, 3, 4, 5)

        np_out = np.minimum(np.maximum(0, arr), 6.0)
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestTanhModule(flow.unittest.TestCase):
    def _test_body_tanh(test_case, input_arr):
        x = flow.Tensor(input_arr)

        tanh = flow.nn.Tanh()
        y = tanh(x)
        z = np.tanh(input_arr)

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def _test_ones_body_tanh(self, shape):
        x = np.ones(shape, dtype=np.float32)
        self._test_body_tanh(x)

    def _test_random_body_tanh(self, shape):
        x = np.random.random(shape).astype(np.float32)
        self._test_body_tanh(x)

    def test_ones_input_tanh(self):
        self._test_ones_body_tanh((1))
        self._test_ones_body_tanh((1, 10))
        self._test_ones_body_tanh((2, 10, 2))
        self._test_ones_body_tanh((2, 5, 2, 2))

    def test_random_input_tanh(self):
        self._test_random_body_tanh((1))
        self._test_random_body_tanh((1, 10))
        self._test_random_body_tanh((2, 10, 2))
        self._test_random_body_tanh((2, 5, 2, 2))

    def _test_body_tanh_v2(test_case, input_arr):
        x = flow.Tensor(input_arr)

        y = flow.tanh(x)
        z = np.tanh(input_arr)

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def _test_body_tanh_v3(test_case, input_arr):
        x = flow.Tensor(input_arr)

        y = x.tanh()
        z = np.tanh(input_arr)

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestELUModule(flow.unittest.TestCase):
    def test_elu(test_case):
        m = flow.nn.ELU()
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.where(arr > 0, arr, 1.0 * (np.exp(arr) - 1))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-4, atol=1e-4))

    def test_elu_alpha(test_case):
        m = flow.nn.ELU(alpha=1.2)
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.where(arr > 0, arr, 1.2 * (np.exp(arr) - 1))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-4, atol=1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestGeLU(flow.unittest.TestCase):
    def test_gelu_v1(test_case):
        input_arr = np.array([-0.5, 0, 0.5]).astype(np.float32)
        x = flow.Tensor(input_arr)

        gelu = flow.nn.GELU()
        y = gelu(x)
        z = np.array([-0.15426877, 0.0, 0.34573123])

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def test_gelu_v2(test_case):
        input_arr = np.array([-0.5, 0, 0.5]).astype(np.float32)
        x = flow.Tensor(input_arr)

        y = flow.gelu(x)
        z = np.array([-0.15426877, 0.0, 0.34573123])

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def test_gelu_v3(test_case):
        input_arr = np.array([-0.5, 0, 0.5]).astype(np.float32)
        x = flow.Tensor(input_arr)

        y = x.gelu()

        z = np.array([-0.15426877, 0.0, 0.34573123])

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))


def numpy_sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def numpy_softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def numpy_logsoftmax(x, dim):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return np.log(e_x / e_x.sum(axis=dim, keepdims=True))


def _test_sigmoid(test_case, device):
    m = flow.nn.Sigmoid()
    input_arr = np.random.randn(2, 3, 4, 5)
    x = flow.Tensor(input_arr, device=flow.device(device))

    y = m(x)
    y2 = flow.sigmoid(x)
    y3 = x.sigmoid()
    output = numpy_sigmoid(input_arr)

    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))
    test_case.assertTrue(np.allclose(y2.numpy(), output, rtol=1e-05))
    test_case.assertTrue(np.allclose(y3.numpy(), output, rtol=1e-05))


input_arr = np.array(
        [[[[-0.95641529, -0.33836755,  0.26598063, -0.00823273, -0.80388736],
          [ 0.09240951, -0.75501866, -1.03428711, -2.10613647,  1.05532779],
          [-0.67854187, -0.94780513,  0.24925377, -0.00432167, -0.79694556],
          [ 1.35626020, -0.79822297,  0.25074951,  0.50052381, -0.78390856]],

         [[-0.19396977, -1.38873898, -0.17250031,  1.09584873, -2.10629595],
          [ 0.34353557,  0.17808227, -1.39449640,  0.88100152,  1.86778366],
          [-0.45035600,  0.14212714, -1.22472810, -0.57314538, -0.64074900],
          [-0.18774252,  0.00437360,  0.67405664,  0.74005017, -0.17124877]],

         [[-1.36008557,  1.47017247,  2.05479638,  0.91677679,  1.96772222],
          [ 0.27401521,  0.26896736, -1.46770821, -1.36479514, -0.11881949],
          [-0.63072382,  1.34082520, -0.95297480, -0.32014565, -0.07749567],
          [-0.55809006, -0.96415186,  0.70116663,  0.31967470,  1.10657454]]],


        [[[ 0.85289628,  0.14198013, -2.31979308,  0.94261968, -0.23640279],
          [-0.01019392,  0.96381344, -0.54525402,  0.98107156,  0.59878900],
          [ 0.93811581,  0.80909467, -1.41894316,  0.98287681,  1.30453298],
          [ 0.59916493,  0.44785232, -1.33960928,  0.68958683,  0.93302651]],

         [[-0.36935154,  0.16810594, -0.23613826,  0.65878148, -0.81449498],
          [ 1.36965517, -0.59050731, -1.20143276,  1.12391894, -0.15764997],
          [-0.57286879,  0.32637867,  1.58773895,  0.34880018,  0.27816965],
          [-0.70959877, -0.49670752, -0.33882379,  0.94732105,  0.21892354]],

         [[-0.93099061, -0.08200545, -0.21850582, -0.17287183, -0.10378965],
          [-0.58051805,  1.00709978, -0.50816052,  0.79838665,  1.07060019],
          [-0.15948598, -0.73000249,  0.23612625, -0.05201063,  0.74377842],
          [ 1.40390348, -0.43790416,  0.84728919, -0.40448909, -0.91392069]]]]
    )

def _test_sigmoid_backward(test_case, device):   
    x_grad = np.array(
        [[[[0.20053668, 0.24297858, 0.24563001, 0.24999576, 0.21359330],
          [0.24946704, 0.21750227, 0.19347675, 0.09672917, 0.19153437],
          [0.22329613, 0.20130318, 0.24615689, 0.24999883, 0.21415766],
          [0.16288576, 0.21405403, 0.24611111, 0.23497355, 0.21520959]],

         [[0.24766315, 0.15976534, 0.24814941, 0.18775899, 0.09671710],
          [0.24276665, 0.24802835, 0.15921283, 0.20713870, 0.11589637],
          [0.23774022, 0.24874173, 0.17552857, 0.23054292, 0.22599894],
          [0.24780992, 0.24999880, 0.22362270, 0.21866858, 0.24817604]],

         [[0.16251797, 0.15197866, 0.10067080, 0.20403911, 0.10759472],
          [0.24536534, 0.24553250, 0.15221321, 0.16206526, 0.24911969],
          [0.22669689, 0.16437025, 0.20084333, 0.24370203, 0.24962503],
          [0.23150114, 0.19984536, 0.22162582, 0.24372023, 0.18675281]]],


        [[[0.20952888, 0.24874432, 0.08148722, 0.20176331, 0.24653939],
          [0.24999351, 0.19987565, 0.23230201, 0.19832526, 0.22886489],
          [0.20216204, 0.21316804, 0.15686963, 0.19816243, 0.16786222],
          [0.22883987, 0.23787172, 0.16448722, 0.22248548, 0.20261154]],

         [[0.24166389, 0.24824206, 0.24654705, 0.22472326, 0.21272532],
          [0.16159818, 0.22941305, 0.17775754, 0.18512031, 0.24845307],
          [0.23056071, 0.24345875, 0.14090340, 0.24254772, 0.24522554],
          [0.22099365, 0.23519269, 0.24295999, 0.20134618, 0.24702830]],

         [[0.20279105, 0.24958016, 0.24703954, 0.24814147, 0.24932794],
          [0.23006634, 0.19596598, 0.23453082, 0.21404074, 0.19011651],
          [0.24841698, 0.21944293, 0.24654740, 0.24983101, 0.21837950],
          [0.15831060, 0.23838789, 0.21000073, 0.24004680, 0.20428880]]]]
    )
    x = flow.Tensor(input_arr, device=flow.device(device), requires_grad=True)
    m = flow.nn.Sigmoid() 
    y = m(x)
    y.retain_grad()
    z = y.sum()
    z.backward()
    test_case.assertTrue(np.allclose(x.grad.numpy(), x_grad, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSigmoid(flow.unittest.TestCase):
    def test_sigmoid(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_sigmoid,
            _test_sigmoid_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


def _test_softmax(test_case, device):
    axis = 0
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(2, 3, 4, 5)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_softmax_dim_1(test_case, device):
    axis = 1
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(9, 7, 8, 16)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_softmax_dim_2(test_case, device):
    axis = 2
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(2, 5, 6, 3)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_softmax_dim_3(test_case, device):
    axis = 3
    m = flow.nn.Softmax(dim=axis)
    arr = np.random.randn(1, 3, 4, 7)
    x = flow.Tensor(arr, device=flow.device(device))
    y = m(x)
    output = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

    axis2 = -1
    m2 = flow.nn.Softmax(dim=axis)
    y2 = m(x)
    output2 = numpy_softmax(arr, axis)
    test_case.assertTrue(np.allclose(y2.numpy(), output2, rtol=1e-05))


def _test_softmax_backward(test_case, device):
    axis = 0
    m = flow.nn.Softmax(dim=axis)
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    y = m(x)
    y.retain_grad()
    z = y.sum()
    z.backward()
    test_case.assertTrue(np.allclose(y.grad.numpy(), np.ones((2, 3, 4, 5)), 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestHardsigmoidModule(flow.unittest.TestCase):
    def test_hardsigmoid(test_case):
        m = flow.nn.Hardsigmoid()
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.maximum(0, np.minimum(1, (arr + 3) / 6))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSoftmax(flow.unittest.TestCase):
    def test_softmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_softmax,
            _test_softmax_dim_1,
            _test_softmax_dim_2,
            _test_softmax_dim_3,
            _test_softmax_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


def _test_logsoftmax(test_case, device):
    dim = 1
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(4, 7)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_logsoftmax_dim_2(test_case, device):
    dim = 2
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(3, 4, 5)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_logsoftmax_dim_3(test_case, device):
    dim = 3
    m = flow.nn.LogSoftmax(dim)
    input_arr = np.random.randn(8, 9, 7, 3)
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    output = numpy_logsoftmax(input_arr, dim)
    test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


def _test_logsoftmax_backward(test_case, device):
    axis = 0
    m = flow.nn.LogSoftmax(axis)
    arr = np.random.randn(2, 3, 4, 5)
    x = flow.Tensor(arr, requires_grad=True, device=flow.device(device))
    y = m(x)
    y.retain_grad()
    z = y.sum()
    z.backward()
    test_case.assertTrue(np.allclose(y.grad.numpy(), np.ones((2, 3, 4, 5)), 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLogSoftmax(flow.unittest.TestCase):
    def test_log_softmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [
            _test_logsoftmax,
            _test_logsoftmax_dim_2,
            _test_logsoftmax_dim_3,
            _test_logsoftmax_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestHardswishModule(flow.unittest.TestCase):
    def test_hardswish(test_case):
        m = flow.nn.Hardswish()
        arr = np.random.randn(2, 3, 4, 5)
        f = arr + 3
        relu6 = np.where(np.where(f < 0, 0, f) > 6, 6, np.where(f < 0, 0, f))
        np_out = arr * relu6 / 6
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestHardtanhModule(flow.unittest.TestCase):
    def test_hardtanh(test_case):
        m = flow.nn.Hardtanh()
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.maximum(-1, np.minimum(1, arr))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))

    def test_hardtanh_min_max(test_case):
        m = flow.nn.Hardtanh(min_val=-2.0, max_val=2.3)
        arr = np.random.randn(2, 3, 4, 5)
        np_out = np.maximum(-2.0, np.minimum(2.3, arr))
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLeakyReLUModule(flow.unittest.TestCase):
    def test_leaky_relu(test_case):
        negative_slope = 0.2
        m = flow.nn.LeakyReLU(negative_slope=negative_slope)
        arr = np.random.randn(2, 3, 4, 5)

        np_out = np.maximum(0, arr) + negative_slope * np.minimum(0, arr)
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


if __name__ == "__main__":
    unittest.main()

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
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest


def _test_convtranspose1d_bias_false(test_case, device):
    np_arr = np.array([[[0.35356437, -0.95761778, 0.19567713]]])
    weight = np.ones((1, 2, 3))
    test_out_data = np.array(
        [
            [
                [0.35356438, -0.6040534, -0.40837622, -0.7619406, 0.19567713],
                [0.35356438, -0.6040534, -0.40837622, -0.7619406, 0.19567713],
            ]
        ]
    )
    test_out_grad = np.array([[[6.0, 6.0, 6.0]]])
    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = nn.ConvTranspose1d(1, 2, 3, stride=1, bias=False)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)
    test_case.assertTrue(np.allclose(out_flow.numpy(), test_out_data, 1e-03, 1e-05))

    out_flow = out_flow.sum()
    out_flow.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), test_out_grad, 1e-06, 1e-06)
    )


def _test_convtranspose1d_bias_true(test_case, device):
    np_arr = np.array([[[0.54925832, -0.64144184, 0.15213189]]])
    weight = np.ones((1, 2, 3))
    bias = np.array([0.16849578, 0.1509564])
    test_out_data = np.array(
        [
            [
                [0.71775407, 0.07631224, 0.22844413, -0.32081416, 0.32062766],
                [0.7002147, 0.05877288, 0.21090476, -0.3383535, 0.3030883],
            ]
        ]
    )
    test_out_grad = np.array([[[6.0, 6.0, 6.0]]])

    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = nn.ConvTranspose1d(1, 2, 3, stride=1, bias=True)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f.bias = nn.Parameter(flow.Tensor(bias))
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)
    test_case.assertTrue(np.allclose(out_flow.numpy(), test_out_data, 1e-02, 1e-05))
    out_flow = out_flow.sum()
    out_flow.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), test_out_grad, 1e-06, 1e-06)
    )


def _test_convtranspose1d_group_bias_false(test_case, device):
    np_arr = np.array(
        [[[0.38072484, -0.01421228, -0.6512485], [-0.05744093, 2.47079971, 0.17573214]]]
    )
    weight = np.ones((2, 1, 3))
    test_out_data = np.array(
        [
            [
                [0.38072485, 0.36651257, -0.28473592, -0.66546077, -0.6512485],
                [-0.05744093, 2.4133587, 2.5890908, 2.6465318, 0.17573214],
            ]
        ]
    )
    test_out_grad = np.array([[[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]])
    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = nn.ConvTranspose1d(2, 2, 3, stride=1, groups=2, bias=False)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)
    test_case.assertTrue(np.allclose(out_flow.numpy(), test_out_data, 1e-06, 1e-06))
    out_flow = out_flow.sum()
    out_flow.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), test_out_grad, 1e-06, 1e-06)
    )


def _test_convtranspose1d_group_bias_true(test_case, device):
    np_arr = np.array(
        [
            [
                [-0.77808793, 0.99824008, 0.57340066],
                [1.46278707, -0.65234252, -1.13087643],
            ],
            [
                [0.76053973, 0.62332447, -1.17157106],
                [0.60291466, -0.0472167, 0.89986403],
            ],
        ]
    )
    weight = np.ones((2, 1, 3))
    bias = np.array([0.32546719, 0.14995032])
    test_out_data = np.array(
        [
            [
                [-0.45262071, 0.54561937, 1.11902, 1.897108, 0.89886785],
                [1.6127374, 0.96039486, -0.1704815, -1.6332686, -0.9809261],
            ],
            [
                [1.0860069, 1.7093314, 0.5377604, -0.22277936, -0.8461038],
                [0.75286496, 0.70564824, 1.6055121, 1.0025976, 1.0498143],
            ],
        ]
    )
    test_out_grad = np.array(
        [[[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]]
    )
    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = nn.ConvTranspose1d(2, 2, 3, stride=1, groups=2, bias=True)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f.bias = nn.Parameter(flow.Tensor(bias))
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)
    test_case.assertTrue(np.allclose(out_flow.numpy(), test_out_data, 1e-06, 1e-06))
    out_flow = out_flow.sum()
    out_flow.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), test_out_grad, 1e-06, 1e-06)
    )


def _test_convtranspose1d_group_large_out_channel(test_case, device):
    np_arr = np.array(
        [
            [
                [2.00934643, 1.5782626, -1.59060988],
                [-1.70463546, 1.30170714, -1.04025804],
            ],
            [
                [0.60327536, 1.26085986, -0.58499662],
                [-0.48145872, -1.64391469, -0.09332249],
            ],
        ]
    )
    weight = np.ones((2, 3, 3))
    test_out_data = np.array(
        [
            [
                [2.0093465, 3.587609, 1.9969991, -0.01234734, -1.5906099],
                [2.0093465, 3.587609, 1.9969991, -0.01234734, -1.5906099],
                [2.0093465, 3.587609, 1.9969991, -0.01234734, -1.5906099],
                [-1.7046355, -0.40292835, -1.4431864, 0.2614491, -1.040258],
                [-1.7046355, -0.40292835, -1.4431864, 0.2614491, -1.040258],
                [-1.7046355, -0.40292835, -1.4431864, 0.2614491, -1.040258],
            ],
            [
                [0.60327536, 1.8641353, 1.2791386, 0.6758632, -0.58499664],
                [0.60327536, 1.8641353, 1.2791386, 0.6758632, -0.58499664],
                [0.60327536, 1.8641353, 1.2791386, 0.6758632, -0.58499664],
                [-0.48145872, -2.1253734, -2.2186959, -1.7372372, -0.09332249],
                [-0.48145872, -2.1253734, -2.2186959, -1.7372372, -0.09332249],
                [-0.48145872, -2.1253734, -2.2186959, -1.7372372, -0.09332249],
            ],
        ]
    )
    test_out_grad = np.array(
        [[[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]], [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]]]
    )
    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = nn.ConvTranspose1d(2, 6, 3, stride=1, groups=2, bias=False)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)
    test_case.assertTrue(np.allclose(out_flow.numpy(), test_out_data, 1e-06, 1e-06))
    out_flow = out_flow.sum()
    out_flow.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), test_out_grad, 1e-06, 1e-06)
    )


def _test_convtranspose1d_group_large_in_channel(test_case, device):
    np_arr = np.array(
        [
            [
                [-0.3939792, -0.34989742, 0.15775536],
                [0.927185, 0.25040535, -1.22738067],
                [-0.2187831, -0.24346108, -0.07109655],
                [-1.55353756, -0.37241986, 0.59579139],
            ],
            [
                [-0.01818884, -1.34408642, 1.31260516],
                [0.52124192, 0.52142919, 1.40499944],
                [0.7410308, 1.93069512, 0.25694943],
                [-0.30531658, 0.24990326, -0.9493729],
            ],
        ]
    )
    weight = np.ones((4, 1, 3))
    test_out_data = np.array(
        [
            [
                [0.5332058, 0.43371373, -0.6359115, -1.1691173, -1.0696253],
                [-1.7723207, -2.3882017, -1.8635068, -0.09118611, 0.52469486],
            ],
            [
                [0.50305307, -0.31960416, 2.3980005, 1.8949474, 2.7176046],
                [0.43571424, 2.6163127, 1.9238893, 1.488175, -0.69242346],
            ],
        ]
    )
    test_out_grad = np.array(
        [
            [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
            [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
        ]
    )
    input_flow = flow.tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    m_f = nn.ConvTranspose1d(4, 2, 3, stride=1, groups=2, bias=False)
    m_f.weight.data = flow.tensor(weight, dtype=flow.float32)
    m_f = m_f.to(device)
    out_flow = m_f(input_flow)
    test_case.assertTrue(np.allclose(out_flow.numpy(), test_out_data, 1e-06, 1e-06))
    out_flow = out_flow.sum()
    out_flow.backward()
    test_case.assertTrue(
        np.allclose(input_flow.grad.numpy(), test_out_grad, 1e-06, 1e-06)
    )


@flow.unittest.skip_unless_1n1d()
class TestConvTranspose(flow.unittest.TestCase):
    def test_ConvTranspose1d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_convtranspose1d_bias_false,
            _test_convtranspose1d_bias_true,
            _test_convtranspose1d_group_bias_false,
            _test_convtranspose1d_group_bias_true,
            _test_convtranspose1d_group_large_out_channel,
            _test_convtranspose1d_group_large_in_channel,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5, rtol=1e-2)
    def test_ConvTranspose1d_(test_case):
        channels = random(1, 6)
        m = torch.nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=random(1, 20),
            kernel_size=random(1, 4),
            stride=random() | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(1, 5) | nothing(),
            groups=random(1, 5) | nothing(),
            padding_mode=constant("zeros") | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=3, dim1=channels).to(device)
        y = m(x)
        return y

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @autotest(n=5)
    def test_deconv1d_group_with_random_data(test_case):
        channels = 720  # lcm(1, 2, 3, 4, 5, 6)
        m = torch.nn.ConvTranspose1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=random(1, 4),
            stride=random() | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(1, 5) | nothing(),
            groups=random(1, 7),
            padding_mode=constant("zeros") | nothing(),
        )
        m.train(random())

        device = random_device()
        m.to(device)
        m.pytorch.to("cuda")
        x = random_tensor(ndim=3, dim1=channels).to(device)
        x.pytorch = x.pytorch.to("cuda")
        y = m(x)
        return y

    @autotest(n=5, rtol=1e-2)
    def test_ConvTranspose3d_(test_case):
        channels = random(1, 2)
        m = torch.nn.ConvTranspose3d(
            in_channels=channels,
            out_channels=random(1, 2),
            kernel_size=random(1, 2),
            stride=random() | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(1, 5) | nothing(),
            groups=1,
            padding_mode=constant("zeros") | nothing(),
        )
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_tensor(ndim=5, dim1=channels).to(device)
        y = m(x)
        return y

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @autotest(n=5)
    def test_deconv3d_group_with_random_data(test_case):
        channels = 120  # lcm(1, 2, 3, 4, 5)
        m = torch.nn.ConvTranspose3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=random(1, 4),
            stride=random() | nothing(),
            padding=random(1, 3).to(int) | nothing(),
            dilation=random(1, 5) | nothing(),
            groups=random(1, 6),
            padding_mode=constant("zeros") | nothing(),
        )
        m.train(random())

        device = random_device()
        m.to(device)
        m.pytorch.to("cuda")
        x = random_tensor(ndim=5, dim1=channels).to(device)
        x.pytorch = x.pytorch.to("cuda")
        y = m(x)
        return y

    @autotest(n=3, auto_backward=False)
    @unittest.skip("TODO: functional_conv_transpose might output incorrect result")
    def test_functional_conv_transpose1d(test_case):
        device = random_device()
        channels = random(1, 6)
        img = random_tensor(ndim=3, dim1=channels).to(device)
        kernel = random_tensor(ndim=3, dim0=channels).to(device)
        y = torch.nn.functional.conv_transpose1d(img, kernel)
        return y

    @autotest(n=3, auto_backward=False)
    @unittest.skip("TODO: functional_conv_transpose might output incorrect result")
    def test_functional_conv_transpose2d(test_case):
        device = random_device()
        channels = random(1, 6)
        img = random_tensor(ndim=4, dim1=channels).to(device)
        kernel = random_tensor(ndim=4, dim0=channels).to(device)
        y = torch.nn.functional.conv_transpose2d(img, kernel)
        return y

    @autotest(n=3, auto_backward=False)
    @unittest.skip("TODO: functional_conv_transpose might output incorrect result")
    def test_functional_conv_transpose3d(test_case):
        device = random_device()
        channels = random(1, 6)
        img = random_tensor(ndim=5, dim1=channels).to(device)
        kernel = random_tensor(ndim=5, dim0=channels).to(device)
        y = torch.nn.functional.conv_transpose3d(img, kernel)
        return y


if __name__ == "__main__":
    unittest.main()

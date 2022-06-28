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


def build_module(act_type):
    if act_type == "relu":
        return torch.nn.ReLU()
    elif act_type == "relu6":
        return torch.nn.ReLU6()
    elif act_type == "tanh":
        return torch.nn.Tanh()
    elif act_type == "elu":
        return torch.nn.ELU(alpha=random())
    elif act_type == "celu":
        return torch.nn.CELU(alpha=random())
    elif act_type == "gelu":
        return torch.nn.GELU()
    elif act_type == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_type == "hardsigmoid":
        return torch.nn.Hardsigmoid()
    elif act_type == "hardshrink":
        return torch.nn.Hardshrink(lambd=random())
    elif act_type == "logsigmoid":
        return torch.nn.LogSigmoid()
    elif act_type == "hardswish":
        return torch.nn.Hardswish()
    elif act_type == "hardtanh":
        return torch.nn.Hardtanh(
            min_val=random().to(float), max_val=random().to(float),
        )
    elif act_type == "leakyrelu":
        return torch.nn.LeakyReLU(negative_slope=random())
    elif act_type == "mish":
        return torch.nn.Mish()
    elif act_type == "silu":
        return torch.nn.SiLU()
    elif act_type == "selu":
        return torch.nn.SELU()
    elif act_type == "threshold":
        return torch.nn.Threshold(threshold=random(), value=random())
    elif act_type == "softplus":
        return torch.nn.Softplus()
    elif act_type == "softshrink":
        return torch.nn.Softshrink()
    else:
        raise ValueError("activation type %s is not support" % act_type)


@autotest(n=1, check_graph=False)
def _test_activation_module_with_random_data(test_case, act_type, ndim, placement, sbp):
    m = build_module(act_type)
    m.train(random())
    dims = [random(1, 3) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=False)
def _test_activation_module_with_0dim_data(test_case, act_type, placement, sbp):
    m = build_module(act_type)
    m.train(random())
    x = random_tensor(ndim=0).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=False)
def _test_activation_module_with_0_size_data(
    test_case, act_type, ndim, zerodim, placement, sbp
):
    m = build_module(act_type)
    m.train(random())
    dims = [random(1, 3) * 8 for i in range(ndim)]
    dims[zerodim] = 0
    x = random_tensor(ndim, *dims).to_global(placement=placement, sbp=sbp)
    y = m(x)
    return y


@globaltest
def _test_activation_module(test_case, act_type):
    for placement in all_placement():
        ndim = random(1, 4).to(int).value()
        for sbp in all_sbp(placement, max_dim=ndim):
            _test_activation_module_with_random_data(
                test_case, act_type, ndim, placement, sbp
            )
        # Skip gelu 0 size test since "Floating point exception" maybe encountered in PyTorch.
        if act_type != "gelu":
            zerodim = random(0, ndim).to(int).value()
            valid_split_axis = [i for i in range(ndim) if i != zerodim]
            for sbp in all_sbp(
                placement, max_dim=ndim, valid_split_axis=valid_split_axis
            ):
                _test_activation_module_with_0_size_data(
                    test_case, act_type, ndim, zerodim, placement, sbp
                )
        for sbp in all_sbp(placement, max_dim=0):
            _test_activation_module_with_0dim_data(test_case, act_type, placement, sbp)


class TestReLUModule(flow.unittest.TestCase):
    def test_relu_module(test_case):
        _test_activation_module(test_case, "relu")


class TestReLU6Module(flow.unittest.TestCase):
    def test_relu6_module(test_case):
        _test_activation_module(test_case, "relu6")


class TestTanh(flow.unittest.TestCase):
    def test_tanh_module(test_case):
        _test_activation_module(test_case, "tanh")


class TestELUModule(flow.unittest.TestCase):
    def test_elu_module(test_case):
        _test_activation_module(test_case, "elu")


class TestCELUModule(flow.unittest.TestCase):
    def test_celu_module(test_case):
        _test_activation_module(test_case, "celu")


class TestGelu(flow.unittest.TestCase):
    def test_gelu_module(test_case):
        _test_activation_module(test_case, "gelu")


class TestSigmoidModule(flow.unittest.TestCase):
    def test_sigmoid_module(test_case):
        _test_activation_module(test_case, "sigmoid")


class TestHardsigmoidModule(flow.unittest.TestCase):
    def test_hardsigmoid_module(test_case):
        _test_activation_module(test_case, "hardsigmoid")


class TestHardshrinkModule(flow.unittest.TestCase):
    def test_hardshrink_module(test_case):
        _test_activation_module(test_case, "hardshrink")


class TestLogSigmoidModule(flow.unittest.TestCase):
    def test_logsigmoid_module(test_case):
        _test_activation_module(test_case, "logsigmoid")


class TestHardswishModule(flow.unittest.TestCase):
    def test_hardswish_module(test_case):
        _test_activation_module(test_case, "hardswish")


class TestHardtanhModule(flow.unittest.TestCase):
    def test_hardtanh_module(test_case):
        _test_activation_module(test_case, "hardtanh")


class TestLeakyReLUModule(flow.unittest.TestCase):
    def test_leakyrelu_module(test_case):
        _test_activation_module(test_case, "leakyrelu")


class TestMishModule(flow.unittest.TestCase):
    def test_mish_module(test_case):
        _test_activation_module(test_case, "mish")


class TestSiluModule(flow.unittest.TestCase):
    def test_silu_module(test_case):
        _test_activation_module(test_case, "silu")


class TestSeluModule(flow.unittest.TestCase):
    def test_selu_module(test_case):
        _test_activation_module(test_case, "selu")


class TestThresholdModule(flow.unittest.TestCase):
    def test_threshold_module(test_case):
        _test_activation_module(test_case, "threshold")


class TestSoftplusModule(flow.unittest.TestCase):
    def test_softplus_module(test_case):
        _test_activation_module(test_case, "softplus")


class TestSoftshrinkModule(flow.unittest.TestCase):
    def test_softshrink_module(test_case):
        _test_activation_module(test_case, "softshrink")


if __name__ == "__main__":
    unittest.main()

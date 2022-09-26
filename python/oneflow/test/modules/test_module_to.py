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
import os
import unittest
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

dummy_val = np.random.randn(2, 3)
in_val = np.full((2, 3), -2)
cpu0_device = flow.device("cpu")
if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
    gpu0_device = cpu0_device
else:
    gpu0_device = flow.device("cuda")


class DummyModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("dummy_buf", flow.Tensor(dummy_val))
        self.dummy_para = flow.nn.Parameter(flow.Tensor(dummy_val))
        self.dummy_para_int = flow.nn.Parameter(flow.Tensor(dummy_val).to(flow.int32))

    def forward(self, x):
        return self.dummy_para * x + self.dummy_buf


def _test_dummy_module(test_case):
    m = DummyModule()
    test_case.assertEqual(m.dummy_buf.device, cpu0_device)
    test_case.assertEqual(m.dummy_para.device, cpu0_device)
    input = flow.Tensor(in_val)
    output = m(input)
    test_case.assertTrue(np.allclose(output.numpy(), -dummy_val, 0.0001, 0.0001))
    test_case.assertEqual(m.dummy_buf.grad, None)
    test_case.assertEqual(m.dummy_para.grad, None)
    test_case.assertEqual(input.device, cpu0_device)
    test_case.assertEqual(output.device, cpu0_device)


def _test_dummy_module_to(test_case):
    m = DummyModule()
    test_case.assertEqual(m.dummy_buf.device, cpu0_device)
    test_case.assertEqual(m.dummy_para.device, cpu0_device)
    m.to(gpu0_device)
    test_case.assertEqual(m.dummy_buf.device, gpu0_device)
    test_case.assertTrue(m.dummy_buf.is_leaf)
    test_case.assertTrue(not m.dummy_buf.requires_grad)
    test_case.assertEqual(m.dummy_para.device, gpu0_device)
    test_case.assertTrue(m.dummy_para.is_leaf)
    test_case.assertTrue(m.dummy_para.requires_grad)
    input = flow.Tensor(in_val).to(gpu0_device)
    output = m(input)
    test_case.assertTrue(np.allclose(output.numpy(), -dummy_val, 0.0001, 0.0001))
    test_case.assertEqual(m.dummy_buf.grad, None)
    test_case.assertEqual(m.dummy_para.grad, None)
    test_case.assertEqual(input.device, gpu0_device)
    test_case.assertEqual(output.device, gpu0_device)
    output_grad = flow.ones((2, 3)).to(gpu0_device)
    output.backward(output_grad)
    test_case.assertEqual(output_grad.device, gpu0_device)
    test_case.assertEqual(m.dummy_buf.grad, None)
    test_case.assertTrue(np.allclose(m.dummy_para.grad.numpy(), in_val, 0.0001, 0.0001))
    test_case.assertEqual(m.dummy_para.grad.device, gpu0_device)


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestModuleTo(flow.unittest.TestCase):
    def test_module_to_device(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_dummy_module, _test_dummy_module_to]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_module_to_dtype(test_case):
        m = DummyModule()
        m.to(flow.float64)
        test_case.assertEqual(m.dummy_buf.dtype, flow.float64)
        test_case.assertEqual(m.dummy_para.dtype, flow.float64)
        test_case.assertEqual(m.dummy_para_int.dtype, flow.int32)

    def test_module_to_tensor(test_case):
        m = DummyModule()
        m.to(flow.zeros(1, dtype=flow.float16, device="cuda"))
        test_case.assertEqual(m.dummy_buf.dtype, flow.float16)
        test_case.assertEqual(m.dummy_para.dtype, flow.float16)
        test_case.assertEqual(m.dummy_para_int.dtype, flow.int32)
        test_case.assertEqual(m.dummy_buf.device.type, "cuda")
        test_case.assertEqual(m.dummy_para.device.type, "cuda")
        test_case.assertEqual(m.dummy_para_int.device.type, "cuda")

    def test_module_to_with_var_reuse(test_case):
        class ReuseVarModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = flow.nn.Linear(3, 4)
                self.linear2 = flow.nn.Linear(3, 4)
                self.linear2.weight = self.linear1.weight

        reuse_var_m = ReuseVarModule()

        test_case.assertTrue(reuse_var_m.linear1.weight is reuse_var_m.linear2.weight)
        test_case.assertEqual(reuse_var_m.linear1.weight.device, cpu0_device)

        test_case.assertTrue(reuse_var_m.linear1.bias is not reuse_var_m.linear2.bias)
        test_case.assertEqual(reuse_var_m.linear1.bias.device, cpu0_device)

        reuse_var_m.to(gpu0_device)

        test_case.assertTrue(reuse_var_m.linear1.weight is reuse_var_m.linear2.weight)
        test_case.assertEqual(reuse_var_m.linear1.weight.device, gpu0_device)

        test_case.assertTrue(reuse_var_m.linear1.bias is not reuse_var_m.linear2.bias)
        test_case.assertEqual(reuse_var_m.linear1.bias.device, gpu0_device)


if __name__ == "__main__":
    unittest.main()

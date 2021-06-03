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
import numpy as np

from oneflow.python.nn.module import Module
import oneflow.experimental as flow

dummy_val = np.random.randn(2, 3)
in_val = np.full((2, 3), -2)
cpu0_device = flow.device("cpu")
gpu0_device = flow.device("cuda")


class DummyModule(Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("dummy_buf", flow.Tensor(dummy_val))
        self.dummy_para = flow.nn.Parameter(flow.Tensor(dummy_val))

    def forward(self, x):
        return (self.dummy_para * x) + self.dummy_buf


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestModuleTo(flow.unittest.TestCase):
    def test_dummy_module(test_case):
        m = DummyModule()

        test_case.assertEqual(m.dummy_buf.device, cpu0_device)
        test_case.assertEqual(m.dummy_para.device, cpu0_device)

        input = flow.Tensor(in_val)
        output = m(input)

        test_case.assertTrue(np.allclose(output.numpy(), -dummy_val, 1e-4, 1e-4))
        test_case.assertEqual(m.dummy_buf.grad, None)
        test_case.assertEqual(m.dummy_para.grad, None)
        test_case.assertEqual(input.device, cpu0_device)
        test_case.assertEqual(output.device, cpu0_device)

    def test_dummy_module_to(test_case):
        m = DummyModule()

        test_case.assertEqual(m.dummy_buf.device, cpu0_device)
        test_case.assertEqual(m.dummy_para.device, cpu0_device)

        # test to
        m.to(gpu0_device)
        test_case.assertEqual(m.dummy_buf.device, gpu0_device)
        test_case.assertTrue(m.dummy_buf.is_leaf)
        test_case.assertTrue(not m.dummy_buf.requires_grad)
        test_case.assertEqual(m.dummy_para.device, gpu0_device)
        test_case.assertTrue(m.dummy_para.is_leaf)
        test_case.assertTrue(m.dummy_para.requires_grad)

        input = flow.Tensor(in_val).to(gpu0_device)
        output = m(input)

        test_case.assertTrue(np.allclose(output.numpy(), -dummy_val, 1e-4, 1e-4))
        test_case.assertEqual(m.dummy_buf.grad, None)
        test_case.assertEqual(m.dummy_para.grad, None)
        test_case.assertEqual(input.device, gpu0_device)
        test_case.assertEqual(output.device, gpu0_device)

        # test to with backward
        output_grad = flow.ones((2, 3)).to(gpu0_device)
        output.backward(output_grad)

        test_case.assertEqual(output_grad.device, gpu0_device)
        test_case.assertEqual(m.dummy_buf.grad, None)
        test_case.assertTrue(np.allclose(m.dummy_para.grad.numpy(), in_val, 1e-4, 1e-4))
        test_case.assertEqual(m.dummy_para.grad.device, gpu0_device)


if __name__ == "__main__":
    unittest.main()

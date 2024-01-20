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
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import torch


def _func_tensor(x):
    return x.exp().sum(dim=1)


def _func_scalar(x):
    return x.exp().sum()


def _func_multi_tensor(x, y):
    return (x.exp() + y.pow(2)).sum(dim=1)


def _func_multi_scalar(x, y):
    return (x.exp() + y.pow(2)).sum()


def _func_scalar2tensor(x):
    return (x, x ** 2, x ** 3)


@flow.unittest.skip_unless_1n1d()
class TestAutogradFunctional(flow.unittest.TestCase):
    def test_vjp(test_case):
        inputs = torch.randn(5, 5)
        v = torch.randn(5)
        result_tensor = torch.autograd.functional.vjp(_func_tensor, inputs, v)
        result_scalar = torch.autograd.functional.vjp(_func_scalar, inputs)

        inputs = (torch.randn(5, 5), torch.randn(5, 5))
        result_tensors = torch.autograd.functional.vjp(_func_multi_tensor, inputs, v)
        result_scalars = torch.autograd.functional.vjp(_func_multi_scalar, inputs)

        return [result_tensor, result_scalar, result_tensors, result_scalars]

    def test_jvp(test_case):
        inputs = torch.randn(5, 5)
        v = torch.randn(5, 5)
        result_tensor = torch.autograd.functional.jvp(_func_tensor, inputs, v)

        inputs = (torch.randn(5, 5), torch.randn(5, 5))
        v = (torch.randn(5, 5), torch.randn(5, 5))
        result_tensors = torch.autograd.functional.jvp(_func_multi_tensor, inputs, v)

        inputs = torch.randn(1)
        result_scalar2tensor = torch.autograd.functional.jvp(
            _func_scalar2tensor, inputs
        )

        return [result_tensor, result_tensors, result_scalar2tensor]

    def test_vhp(test_case):
        inputs = torch.randn(5, 5)
        v = torch.randn(5, 5)
        result_tensor = torch.autograd.functional.vhp(_func_scalar, inputs, v)

        inputs = (torch.randn(5, 5), torch.randn(5, 5))
        v = (torch.randn(5, 5), torch.randn(5, 5))
        result_tensors = torch.autograd.functional.vhp(_func_multi_scalar, inputs, v)

        return [result_tensor, result_tensors]

    def test_hvp(test_case):
        inputs = torch.randn(5, 5)
        v = torch.randn(5, 5)
        result_tensor = torch.autograd.functional.hvp(_func_scalar, inputs, v)

        inputs = (torch.randn(5, 5), torch.randn(5, 5))
        v = (torch.randn(5, 5), torch.randn(5, 5))
        result_tensors = torch.autograd.functional.hvp(_func_multi_scalar, inputs, v)

        return [result_tensor, result_tensors]

    def test_jacobian(test_case):
        inputs = torch.randn(5, 5)
        result_tensor = torch.autograd.functional.jacobian(
            _func_tensor, inputs, vectorize=False, strategy="reverse-mode"
        )

        inputs = (torch.randn(5, 5), torch.randn(5, 5))
        result_tensors = torch.autograd.functional.jacobian(
            _func_multi_scalar, inputs, vectorize=False, strategy="reverse-mode"
        )

        return [result_tensor, result_tensors]

    def test_hessian(test_case):
        inputs = torch.randn(5, 5)
        result_tensor = torch.autograd.functional.hessian(
            _func_scalar,
            inputs,
            vectorize=False,
            outer_jacobian_strategy="reverse-mode",
        )

        inputs = (torch.randn(5, 5), torch.randn(5, 5))
        result_tensors = torch.autograd.functional.hessian(
            _func_multi_scalar,
            inputs,
            vectorize=False,
            outer_jacobian_strategy="reverse-mode",
        )

        return [result_tensor, result_tensors]


if __name__ == "__main__":
    unittest.main()

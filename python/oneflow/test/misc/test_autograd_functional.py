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
import torch


@flow.unittest.skip_unless_1n1d()
class TestAutogradFunctional(flow.unittest.TestCase):
    def test_vjp(test_case):
        def _func_tensor(x):
            return x.exp().sum(dim=1)
        def _func_scalar(x):
            return x.exp().sum()
        inputs = torch.randn(5, 5)
        v = torch.randn(5)
        result_tensor = torch.autograd.functional.vjp(_func_tensor, inputs, v)
        result_scalar = torch.autograd.functional.vjp(_func_scalar, inputs)
        return [result_tensor, result_scalar]

if __name__ == "__main__":
    unittest.main()

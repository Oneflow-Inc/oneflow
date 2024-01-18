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

import numpy as np
import oneflow as flow
import oneflow.unittest
import torch
from oneflow.framework.infer_compiler import compile_from_torch, register
from oneflow.framework.infer_compiler.with_oneflow_compile import (
    DualModule,
    DualModuleList,
)


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestOneflowInferCompiler(flow.unittest.TestCase):
    def setUp(self):
        os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"

    def test_compile_from_torch(test_case):
        class TorchModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = torch.nn.ModuleList(
                    [torch.nn.Linear(10, 10) for i in range(10)]
                )

            def forward(self, x):
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x

        class OneflowModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linears = flow.nn.ModuleList(
                    [flow.nn.Linear(10, 10) for i in range(10)]
                )

            def forward(self, x):
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x

        register(torch2oflow_class_map={TorchModule: OneflowModule})

        m = TorchModule().to("cuda")
        x = torch.randn(2, 10).to("cuda")

        y_torch = m(x)
        m = compile_from_torch(m)
        y_oneflow = m(x)
        test_case.assertTrue(
            np.allclose(y_torch.detach().cpu(), y_oneflow.detach().cpu(), 1e-03, 1e-03)
        )
        test_case.assertIsInstance(m.linears, DualModuleList)

        x = getattr(m.linears, "1")
        test_case.assertIsInstance(x, DualModule)

        x.bias = None
        setattr(m.linears, "2", x)
        test_case.assertIsNone(m.linears[2].bias)
        test_case.assertIsNone(m.linears._torch_modules[2].bias)
        test_case.assertIsNone(m.linears._oneflow_modules[2].bias)

        m.linears[3] = x
        test_case.assertIsNone(m.linears[3].bias)
        test_case.assertIsNone(m.linears._torch_modules[3].bias)
        test_case.assertIsNone(m.linears._oneflow_modules[3].bias)


if __name__ == "__main__":
    unittest.main()

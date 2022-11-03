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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s | FileCheck %s
# CHECK-NOT: oneflow.broadcast_mul

import os
import unittest
import numpy as np
import oneflow as flow
import oneflow.unittest
import oneflow.nn as nn

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"


class MultiplyModel(nn.Module):
    def __init__(self, dtype=flow.float32):
        super().__init__()
        self.dtype = dtype
        self.x = nn.Parameter(flow.tensor([2, 2], dtype=self.dtype), False)
        self.y = nn.Parameter(flow.tensor([3, 3], dtype=self.dtype), False)

    def forward(self):
        return self.x * self.y


class MultiplyModelComplex(MultiplyModel):
    def __init__(self, dtype=flow.float32):
        super().__init__(dtype)
        self.z = nn.Parameter(flow.tensor([4, 5], dtype=self.dtype), False)

    def forward(self):
        return self.x * self.y * self.z


class MultiplyModelWithInput(MultiplyModel):
    def __init__(self, dtype=flow.float32):
        super().__init__(dtype)

    def forward(self, a: flow.Tensor, b: flow.Tensor):
        z = self.x * self.y
        return a + b + z


def _test_fold_multiply(test_case, module, with_cuda, *args, dtype=oneflow.float32):
    model = module(dtype)

    if with_cuda:
        model.to("cuda")
    model.eval()
    eager_res = model(*args)

    class MultiplyGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self, *args):
            return self.model(*args)

    graph = MultiplyGraph()
    lazy_res = graph(*args)

    test_case.assertTrue(
        np.allclose(eager_res.numpy(), lazy_res.numpy(), rtol=1e-5, atol=1e-5)
    )
    test_case.assertTrue(eager_res.dtype == dtype and lazy_res.dtype == dtype)


@flow.unittest.skip_unless_1n1d()
class TestFoldMultiply(oneflow.unittest.TestCase):
    def test_fold_multiply(test_case):
        _test_fold_multiply(test_case, MultiplyModel, with_cuda=False)
        _test_fold_multiply(
            test_case, MultiplyModel, with_cuda=False, dtype=flow.float16
        )

    @unittest.skipUnless(oneflow.sysconfig.with_cuda(), "only test cpu cases")
    def test_fold_multiply_cuda(test_case):
        _test_fold_multiply(test_case, MultiplyModel, with_cuda=True)
        _test_fold_multiply(
            test_case, MultiplyModel, with_cuda=True, dtype=flow.float16
        )

    def test_fold_multiply_complex(test_case):
        _test_fold_multiply(test_case, MultiplyModelComplex, with_cuda=False)
        _test_fold_multiply(
            test_case, MultiplyModelComplex, with_cuda=False, dtype=flow.float16
        )

    @unittest.skipUnless(oneflow.sysconfig.with_cuda(), "only test cpu cases")
    def test_fold_multiply_complex_cuda(test_case):
        _test_fold_multiply(test_case, MultiplyModelComplex, with_cuda=True)
        _test_fold_multiply(
            test_case, MultiplyModelComplex, with_cuda=True, dtype=flow.float16
        )

    def test_fold_multiply_with_input(test_case):
        a = flow.tensor([3, 7], dtype=flow.float32)
        b = flow.tensor([9, -1], dtype=flow.float32)
        a_fp16 = flow.tensor([3, 7], dtype=flow.float16)
        b_fp16 = flow.tensor([9, -1], dtype=flow.float16)
        _test_fold_multiply(test_case, MultiplyModelWithInput, False, a, b)
        _test_fold_multiply(
            test_case, MultiplyModelWithInput, False, a_fp16, b_fp16, dtype=flow.float16
        )

    @unittest.skipUnless(oneflow.sysconfig.with_cuda(), "only test cpu cases")
    def test_fold_multiply_with_input_cuda(test_case):
        a = flow.tensor([3, 7], dtype=flow.float32, device="cuda")
        b = flow.tensor([9, -1], dtype=flow.float32, device="cuda")
        a_fp16 = flow.tensor([3, 7], dtype=flow.float16, device="cuda")
        b_fp16 = flow.tensor([9, -1], dtype=flow.float16, device="cuda")
        _test_fold_multiply(test_case, MultiplyModelWithInput, True, a, b)
        _test_fold_multiply(
            test_case, MultiplyModelWithInput, True, a_fp16, b_fp16, dtype=flow.float16
        )


if __name__ == "__main__":
    unittest.main()

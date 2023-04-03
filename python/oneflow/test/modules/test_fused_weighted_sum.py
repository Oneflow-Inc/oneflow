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
import math
import random
import os

import oneflow as flow


def _ref(inputs, weights, alpha, init_grad, device, dtype):
    inputs = [flow.tensor(t).to(device).to(dtype) for t in inputs]
    for t in inputs:
        t.requires_grad = True
    init_grad = flow.tensor(init_grad).to(device).to(dtype)
    out = inputs[0] * weights[0]
    for i, w in zip(inputs[1:], weights[1:]):
        out += i * w
    out = out * alpha
    out.backward(init_grad)
    return out, [t.grad for t in inputs]


def _fused_weighted_sum(inputs, weights, alpha, init_grad, device, dtype):
    inputs = [flow.tensor(t).to(device).to(dtype) for t in inputs]
    for t in inputs:
        t.requires_grad = True
    init_grad = flow.tensor(init_grad).to(device).to(dtype)
    out = flow._C.fused_weighted_sum(inputs, weights, alpha)
    out.backward(init_grad)
    return out, [t.grad for t in inputs]


def _test_fused_weighted_sum(test_case, shape, n, device, dtype):
    inputs = [np.random.randn(*shape) for _ in range(n)]
    init_grad = np.random.randn(*shape)
    weights = [random.random() for _ in range(n)]
    alpha = random.random()
    out, grads = _fused_weighted_sum(inputs, weights, alpha, init_grad, device, dtype)
    ref, ref_grads = _ref(inputs, weights, alpha, init_grad, device, dtype)
    test_case.assertTrue(np.allclose(ref, out, atol=1e-5, rtol=1e-5))
    for (grad, ref_grad) in zip(grads, ref_grads):
        test_case.assertTrue(np.allclose(ref_grad, grad, atol=1e-5, rtol=1e-5))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestFusedWeightedSum(flow.unittest.TestCase):
    def test_fused_weighted_sum(test_case):
        _test_fused_weighted_sum(test_case, (1024, 1024), 1, "cuda", flow.float32)
        _test_fused_weighted_sum(test_case, (1024, 1024), 3, "cuda", flow.float32)
        _test_fused_weighted_sum(test_case, (1024, 1024), 8, "cuda", flow.float32)
        _test_fused_weighted_sum(test_case, (1024, 1024), 11, "cuda", flow.float32)
        _test_fused_weighted_sum(test_case, (1024, 1024), 21, "cuda", flow.float32)
        _test_fused_weighted_sum(test_case, (1024, 1024), 1, "cpu", flow.float32)
        _test_fused_weighted_sum(test_case, (1024, 1024), 3, "cpu", flow.float32)
        _test_fused_weighted_sum(test_case, (1024, 1024), 8, "cpu", flow.float32)
        _test_fused_weighted_sum(test_case, (1024, 1024), 11, "cpu", flow.float32)
        _test_fused_weighted_sum(test_case, (1024, 1024), 21, "cpu", flow.float32)


if __name__ == "__main__":
    unittest.main()

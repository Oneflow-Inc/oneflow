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
import numpy as np
import unittest
from collections import OrderedDict

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgDict


def _test_fused_fast_gelu_mul(test_case, shape, dtype=flow.float32):
    x = flow.randn(*shape).to(dtype=dtype, device="cuda").requires_grad_(True)
    multiplier = flow.randn(*shape).to(dtype=dtype, device="cuda").requires_grad_(True)
    y = flow.nn.functional.gelu(x, approximate="tanh") * multiplier
    y.mean().backward()
    x_grad = x.grad.detach().cpu()
    m_grad = multiplier.grad.detach().cpu()
    y = y.detach().cpu()

    fused_x = x.detach().clone().requires_grad_(True)
    fused_multiplier = multiplier.detach().clone().requires_grad_(True)
    fused_y = flow._C.fused_fast_gelu_mul(fused_x, fused_multiplier)
    fused_y.mean().backward()
    fused_x_grad = fused_x.grad.detach().cpu()
    fused_m_grad = fused_multiplier.grad.detach().cpu()
    fused_y = fused_y.detach().cpu()

    def compare(a, b, rtol=1e-5, atol=1e-8):
        test_case.assertTrue(
            np.allclose(a.numpy(), b.numpy(), rtol=rtol, atol=atol),
            f"\na\n{a.numpy()}\n{'-' * 80}\nb:\n{b.numpy()}\n{'*' * 80}\ndiff:\n{a.numpy() - b.numpy()}",
        )

    # print(f"\n{'=' * 20} shape={shape} dtype={dtype} {'=' * 20}")
    if dtype == flow.float16:
        compare(fused_y, y, 1e-2, 1e-3)
        compare(fused_x_grad, x_grad, 1e-4, 1e-3)
        compare(fused_m_grad, m_grad, 1e-4, 1e-3)
    else:
        compare(fused_y, y)
        compare(fused_x_grad, x_grad)
        compare(fused_m_grad, m_grad)


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestFusedFastGeluMul(flow.unittest.TestCase):
    def test_fused_fast_gelu_mul(test_case):
        args_dict = OrderedDict()
        args_dict["shape"] = [[5], [7, 10], [4, 2, 3], [8, 3, 16, 16]]
        args_dict["dtype"] = [flow.float16, flow.float32]
        for kwarg in GenArgDict(args_dict):
            _test_fused_fast_gelu_mul(test_case, **kwarg)


if __name__ == "__main__":
    unittest.main()

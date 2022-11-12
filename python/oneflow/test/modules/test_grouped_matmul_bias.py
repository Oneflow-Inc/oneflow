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

import oneflow as flow


def _ref(xs, weights, biases):
    return [
        flow._C.matmul(x, w, transpose_a=False, transpose_b=True) + b
        for x, w, b in zip(xs, weights, biases)
    ]


def _grouped(xs, weights, biases):
    return flow._C.grouped_matmul_bias(xs, weights, biases)


def _test_grouped_matmul_bias(test_case, dtype, problems):

    xs = [flow.randn((m, k), device="cuda", dtype=dtype) for (m, n, k) in problems]
    ws = [flow.randn((n, k), device="cuda", dtype=dtype) for (m, n, k) in problems]
    bs = [flow.randn((n), device="cuda", dtype=dtype) for (m, n, k) in problems]

    ref_out = _ref(xs, ws, bs)
    grouped_out = _grouped(xs, ws, bs)
    for (ref_y, grouped_y) in zip(ref_out, grouped_out):
        test_case.assertTrue(
            np.allclose(ref_y.numpy(), grouped_y.numpy(), atol=1e-2, rtol=1e-2)
        )


@flow.unittest.skip_unless_1n1d()
class TestGroupedMatmulBias(flow.unittest.TestCase):
    def test_grouped_matmul_bias(test_case):
        problems = [(2, 1280, 1280)] * 12 + [(2, 1280, 640)] * 4 + [(2, 1280, 320)] * 5
        _test_grouped_matmul_bias(test_case, flow.float16, problems)
        problems = (
            [(2 * 77, 768, 1280)] * 6
            + [(2 * 77, 768, 640)] * 5
            + [(2 * 77, 768, 320)] * 5
        )
        _test_grouped_matmul_bias(test_case, flow.float16, problems)


if __name__ == "__main__":
    unittest.main()

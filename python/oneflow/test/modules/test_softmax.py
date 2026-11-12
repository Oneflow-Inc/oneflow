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
import os
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _dtype_flow_to_np(dtype):
    return {flow.float32: np.float32, flow.float16: np.float16}[dtype]


def _np_softmax(x, dtype=None):
    if dtype is not None:
        x = x.astype(dtype)
    x -= np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=-1, keepdims=True)


def _test_softmax_impl(test_case, shape, input_dtype, output_dtype):
    np_input = np.random.randn(*shape).astype(_dtype_flow_to_np(input_dtype))
    of_input = flow.tensor(np_input, dtype=input_dtype, device=flow.device("cuda"))
    of_out = flow.nn.functional.softmax(of_input, dtype=output_dtype)
    if output_dtype is not None:
        np_out = _np_softmax(np_input, dtype=_dtype_flow_to_np(output_dtype))
    else:
        np_out = _np_softmax(np_input)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.001, 0.001))


def _test_softmax_wide_rows(test_case, cols, dtype, log):
    np_dtype = _dtype_flow_to_np(dtype)
    rng = np.random.RandomState(cols)
    np_x = rng.randn(cols % 3 + 2, cols).astype(np_dtype)
    np_dy = rng.randn(*np_x.shape).astype(np_dtype)
    x = flow.tensor(np_x, device=flow.device("cuda"), requires_grad=True)
    func = flow.nn.functional.log_softmax if log else flow.nn.functional.softmax
    y = func(x, dim=-1)
    y.backward(flow.tensor(np_dy, device=flow.device("cuda")))

    ref_x = np_x.astype(np.float64)
    dy = np_dy.astype(np.float64)
    p = np.exp(ref_x - np.max(ref_x, axis=-1, keepdims=True))
    p /= np.sum(p, axis=-1, keepdims=True)
    if log:
        ref_y = np.log(p)
        ref_dx = dy - p * np.sum(dy, axis=-1, keepdims=True)
    else:
        ref_y = p
        ref_dx = p * (dy - np.sum(p * dy, axis=-1, keepdims=True))
    tol = 1e-2 if dtype is flow.float16 else 1e-4
    for name, out, ref in [("y", y, ref_y), ("x_grad", x.grad, ref_dx)]:
        out = out.numpy().astype(np.float64)
        atol = tol * np.max(np.abs(ref))
        test_case.assertTrue(
            np.allclose(out, ref, rtol=tol, atol=atol),
            f"{name}: cols={cols} dtype={dtype} log={log} "
            f"max abs diff {np.max(np.abs(out - ref))}",
        )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class Testsoftmax(flow.unittest.TestCase):
    def test_softmax(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(20, 30), (32, 128)]
        arg_dict["input_dtype"] = [flow.float16, flow.float32]
        arg_dict["output_dtype"] = [None, flow.float32]
        for arg in GenArgList(arg_dict):
            _test_softmax_impl(test_case, *arg)

    def test_softmax_wide_rows(test_case):
        # Widths past 8192 on both sides of the row sizes at which a cached row needs 64 KiB
        # and 128 KiB of shared memory, forward and backward.
        widths = [8180, 8190, 12000, 16000, 16350, 16384, 16400, 24000, 32000]
        widths += [32760, 32768, 32800, 40000, 65000]
        for dtype in [flow.float32, flow.float16]:
            for log in [False, True]:
                for cols in widths:
                    _test_softmax_wide_rows(test_case, cols, dtype, log)


if __name__ == "__main__":
    unittest.main()

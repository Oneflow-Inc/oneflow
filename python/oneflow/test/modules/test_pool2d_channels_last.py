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
import torch

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList

_tolerance = {
    flow.float64: 1e-12,
    flow.float32: 1e-6,
    flow.float16: 1e-2,
    flow.bfloat16: 4e-2,
}


def _test_pool2d_channels_last(test_case, kind, dtype, shape, kernel_size, stride, padding):
    rng = np.random.RandomState(sum(shape))
    if kind == "max":
        # Distinct values with 8 significant bits: exact in every tested type and free of ties,
        # so the outputs and the routed gradients must match exactly.
        mantissa, exponent, sign = np.meshgrid(
            1 + np.arange(128) / 128.0, 2.0 ** np.arange(-6, 7), [-1.0, 1.0]
        )
        values = (mantissa * exponent * sign).ravel()
        np_x = rng.choice(values, size=int(np.prod(shape)), replace=False).reshape(shape)
    else:
        # Quarter integers in [-4, 4) are exact in every tested type.
        np_x = rng.randint(-16, 16, size=shape) / 4.0
    x = flow.tensor(np_x, device="cuda", dtype=dtype, requires_grad=True)
    torch_x = torch.tensor(np_x.transpose(0, 3, 1, 2), requires_grad=True)
    if kind == "max":
        y = flow._C.max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=(1, 1),
            return_indices=False,
            ceil_mode=False,
            data_format="channels_last",
        )[0]
        torch_y = torch.nn.functional.max_pool2d(
            torch_x, kernel_size, stride=stride, padding=padding
        )
    else:
        y = flow._C.avg_pool2d(
            x,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=False,
            count_include_pad=True,
            divisor_override=0,
            data_format="channels_last",
        )
        torch_y = torch.nn.functional.avg_pool2d(
            torch_x, kernel_size, stride=stride, padding=padding, count_include_pad=False
        )
    np_dy = rng.randint(-16, 16, size=tuple(y.shape)) / 4.0
    y.backward(flow.tensor(np_dy, device="cuda", dtype=dtype))
    torch_y.backward(torch.tensor(np_dy.transpose(0, 3, 1, 2)))

    tol = _tolerance[dtype]
    ref_y = torch_y.detach().numpy().transpose(0, 2, 3, 1)
    ref_grad = torch_x.grad.numpy().transpose(0, 2, 3, 1)
    test_case.assertEqual(tuple(y.shape), ref_y.shape)
    out = y.detach().to(flow.float64).cpu().numpy()
    grad = x.grad.to(flow.float64).cpu().numpy()
    if kind == "max":
        test_case.assertTrue(np.array_equal(out, ref_y))
        test_case.assertTrue(np.array_equal(grad, ref_grad))
    else:
        test_case.assertTrue(np.allclose(out, ref_y, rtol=tol, atol=tol))
        test_case.assertTrue(np.allclose(grad, ref_grad, rtol=tol, atol=tol))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cuda cases")
@flow.unittest.skip_unless_1n1d()
class TestPool2dChannelsLast(flow.unittest.TestCase):
    def test_pool2d_channels_last(test_case):
        arg_dict = OrderedDict()
        arg_dict["kind"] = ["max", "avg"]
        arg_dict["dtype"] = [flow.float64, flow.float32, flow.float16, flow.bfloat16]
        arg_dict["shape"] = [(2, 8, 8, 3), (2, 13, 11, 4)]
        arg_dict["kernel_size"] = [(3, 3), (2, 3)]
        arg_dict["stride"] = [(1, 1), (2, 1)]
        arg_dict["padding"] = [(0, 0), (1, 1)]
        for arg in GenArgList(arg_dict):
            _test_pool2d_channels_last(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

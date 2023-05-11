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
import os

import oneflow as flow


def _pack_int8_to_int4(x):
    np_x = x.numpy()
    l = np_x[..., 0::2]
    r = np_x[..., 1::2]
    l = np.left_shift(l, 4)
    if x.dtype is flow.int8:
        r = np.bitwise_and(r, np.int8(0xF))
    packed = flow.tensor(np.bitwise_or(l, r), device=x.device)
    return packed


def _unpack_int4_to_int8(x):
    np_x = x.numpy()
    l = np.right_shift(np_x, 4).reshape(x.shape + (1,))
    r = np.right_shift(np.left_shift(np_x, 4), 4).reshape(x.shape + (1,))
    unpacked = np.concatenate((l, r), -1).reshape(x.shape[0:-1] + (x.shape[-1] * 2,))
    unpacked = flow.tensor(unpacked, device=x.device)
    return unpacked


def _quantize(num_bits, symmetric, x, group_dim, group_size, quant_type):
    x_float = x.float()
    x_reshaped = x_float.reshape(
        x.shape[:group_dim]
        + (x.shape[group_dim] // group_size, group_size)
        + x.shape[group_dim + 1 :]
    )
    if symmetric:
        signed_max = float(2 ** (num_bits - 1)) - 1
        offset = signed_max if quant_type is flow.uint8 else 0.0
        scale_float = (
            x_reshaped.abs().max(dim=group_dim + 1, keepdim=True).values / signed_max
        )
        quantized = (
            flow.round(x_reshaped / scale_float + offset)
            .reshape(x.shape)
            .to(quant_type)
        )
        if num_bits == 4:
            quantized = _pack_int8_to_int4(quantized)
        return (quantized, scale_float.squeeze(group_dim + 1).to(x.dtype), None)
    else:
        unsigned_max = float(2 ** num_bits) - 1
        mn = x_reshaped.min(dim=group_dim + 1, keepdim=True).values
        mx = x_reshaped.max(dim=group_dim + 1, keepdim=True).values
        scale_float = (mx - mn) / unsigned_max
        quantized = (
            flow.round((x_reshaped - mn) / scale_float).reshape(x.shape).to(flow.uint8)
        )
        if num_bits == 4:
            quantized = _pack_int8_to_int4(quantized)
        return (
            quantized,
            scale_float.squeeze(group_dim + 1).to(x.dtype),
            mn.squeeze(group_dim + 1).to(x.dtype),
        )


def _dequantize_ref(num_bits, symmetric, quantized, scale, zero, group_dim, group_size):
    if num_bits == 4:
        quantized = _unpack_int4_to_int8(quantized)
    scale_reshaped = scale.unsqueeze(group_dim + 1)
    quantized_reshaped = quantized.reshape(
        quantized.shape[:group_dim]
        + (quantized.shape[group_dim] // group_size, group_size)
        + quantized.shape[group_dim + 1 :]
    )
    if symmetric:
        offset = (
            float(2 ** (num_bits - 1)) - 1 if quantized.dtype is flow.uint8 else 0.0
        )
        dequantized = (quantized_reshaped.to(scale.dtype) - offset) * scale_reshaped
    else:
        zero_reshaped = zero.unsqueeze(group_dim + 1)
        dequantized = (
            zero_reshaped + quantized_reshaped.to(scale.dtype) * scale_reshaped
        )
    return dequantized.reshape(quantized.shape)


def _dequantize(num_bits, symmetric, x, scale, zero, group_dim, group_size):
    return flow._C.groupwise_dequantize(
        x,
        scale=scale,
        zero=zero,
        group_dim=group_dim,
        group_size=group_size,
        num_bits=num_bits,
        symmetric=symmetric,
    )


def _test_dequantize(test_case, num_bits, shape, group_dim, group_size):

    for dtype in [flow.float, flow.float16]:
        x = flow.randn(shape, device="cuda", dtype=flow.float,).to(dtype)
        for symmetric in [True, False]:
            for quant_type in [flow.int8, flow.uint8] if symmetric else [flow.uint8]:
                quantized, scale, zero = _quantize(
                    num_bits, symmetric, x, group_dim, group_size, quant_type
                )
                dequantized = _dequantize(
                    num_bits, symmetric, quantized, scale, zero, group_dim, group_size
                )
                dequantized_ref = _dequantize_ref(
                    num_bits, symmetric, quantized, scale, zero, group_dim, group_size,
                )
                test_case.assertTrue(
                    np.allclose(dequantized_ref, dequantized, atol=1e-2, rtol=1e-2)
                )


def _test_fused_linear(test_case, num_bits, m, k, n, group_dim, group_size):
    for dtype in [flow.float16, flow.float]:
        x = flow.randn((m, k), device="cuda", dtype=flow.float,).to(dtype) / 10
        w = flow.randn((n, k), device="cuda", dtype=flow.float,).to(dtype) / 10
        b = flow.randn((n), device="cuda", dtype=flow.float,).to(dtype) / 10

        for symmetric in [True, False]:
            for quant_type in [flow.int8, flow.uint8] if symmetric else [flow.uint8]:
                w_quantized, w_scale, w_zero = _quantize(
                    num_bits, symmetric, w, group_dim, group_size, quant_type
                )

                fused_out = flow._C.fused_linear_with_groupwise_quantized_weight(
                    x=x,
                    w=w_quantized,
                    w_scale=w_scale,
                    w_zero=w_zero,
                    b=b,
                    num_bits=num_bits,
                    symmetric=symmetric,
                    group_dim=group_dim,
                    group_size=group_size,
                )
                ref = (
                    flow.matmul(
                        x,
                        _dequantize(
                            num_bits,
                            symmetric,
                            w_quantized,
                            w_scale,
                            w_zero,
                            group_dim,
                            group_size,
                        ).t(),
                    )
                    + b
                )

                test_case.assertTrue(np.allclose(ref, fused_out, atol=1e-2, rtol=1e-2))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGroupWiseQuantization(flow.unittest.TestCase):
    def test_dequantize(test_case):
        _test_dequantize(test_case, 8, (128, 256), 0, 128)
        _test_dequantize(test_case, 8, (64, 128, 256), 0, 64)
        _test_dequantize(test_case, 8, (64, 128, 256), 1, 128)
        _test_dequantize(test_case, 8, (64, 128, 256), 2, 256)
        _test_dequantize(test_case, 8, (63, 127, 255), 0, 63)
        _test_dequantize(test_case, 8, (63, 127, 255), 1, 127)
        _test_dequantize(test_case, 8, (63, 127, 255), 2, 255)
        _test_dequantize(test_case, 8, (128, 256), 1, 256 // 4)
        _test_dequantize(test_case, 8, (128, 256), 0, 128 // 4)
        _test_dequantize(test_case, 8, (64, 128, 256), 0, 64 // 4)
        _test_dequantize(test_case, 8, (64, 128, 256), 1, 128 // 4)
        _test_dequantize(test_case, 8, (64, 128, 256), 2, 256 // 4)

        _test_dequantize(test_case, 4, (128, 256), 1, 256)
        _test_dequantize(test_case, 4, (128, 256), 0, 128)
        _test_dequantize(test_case, 4, (64, 128, 256), 0, 64)
        _test_dequantize(test_case, 4, (64, 128, 256), 1, 128)
        _test_dequantize(test_case, 4, (64, 128, 256), 2, 256)
        _test_dequantize(test_case, 4, (128, 256), 1, 256 // 4)
        _test_dequantize(test_case, 4, (128, 256), 0, 128 // 4)
        _test_dequantize(test_case, 4, (64, 128, 256), 0, 64 // 4)
        _test_dequantize(test_case, 4, (64, 128, 256), 1, 128 // 4)
        _test_dequantize(test_case, 4, (64, 128, 256), 2, 256 // 4)

    def test_fused_linear(test_case):
        _test_fused_linear(test_case, 8, 1, 64, 128, 0, 128)
        _test_fused_linear(test_case, 8, 1, 64, 128, 1, 64)
        _test_fused_linear(test_case, 8, 16, 64, 128, 0, 128)
        _test_fused_linear(test_case, 8, 16, 64, 128, 1, 64)
        _test_fused_linear(test_case, 8, 1, 63, 127, 0, 127)
        _test_fused_linear(test_case, 8, 1, 63, 127, 1, 63)
        _test_fused_linear(test_case, 8, 1, 256, 512, 0, 64)
        _test_fused_linear(test_case, 8, 1, 256, 512, 1, 64)
        _test_fused_linear(test_case, 4, 1, 256, 512, 0, 512)
        _test_fused_linear(test_case, 4, 1, 256, 512, 1, 256)
        _test_fused_linear(test_case, 4, 1, 256, 512, 0, 64)
        _test_fused_linear(test_case, 4, 1, 256, 512, 1, 64)


if __name__ == "__main__":
    unittest.main()

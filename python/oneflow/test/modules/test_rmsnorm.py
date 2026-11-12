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
import torch


def _get_norm_dims(shape, normalized_shape):
    lpad = len(shape) - len(normalized_shape)
    assert lpad >= 0
    return tuple(range(lpad, len(shape)))


def _torch_rmsnorm(x, weight, normalized_shape=None, eps=1e-6):
    if weight is not None:
        normalized_shape = weight.shape
    else:
        assert normalized_shape is not None
    norm_dims = _get_norm_dims(x.shape, normalized_shape)
    root_mean = torch.mean(x * x, dim=norm_dims, keepdim=True)
    rms = torch.rsqrt(root_mean + eps)
    normed = x * rms
    return normed * weight if weight is not None else normed


def _warp_dispatch_widths():
    # Every width up to 256, then both sides of each multiple of 32 up to 2048 and a few
    # wider ones, so that each column bucket of the warp and block kernels is reached on
    # 32- and 64-wide warps, for both pack sizes.
    widths = list(range(1, 257))
    for k in range(9, 65):
        widths += [32 * k - 1, 32 * k, 32 * k + 1]
    widths += [3071, 4095, 4096, 4097, 8191, 8192]
    return widths


def _wide_widths():
    # Widths past 8192 on both sides of the row sizes at which a cached row needs 64 KiB and
    # 128 KiB of shared memory, forward and backward, in fp32 and fp16, so that the dispatch
    # from the shared-memory kernels to the uncached ones is reached.
    return [
        8180,
        8190,
        12000,
        16000,
        16350,
        16384,
        16400,
        24000,
        32000,
        32760,
        32768,
        32800,
        40000,
        65000,
    ]


def _test_rmsnorm(
    test_case,
    shape,
    normalized_shape,
    affine=True,
    eps=1e-6,
    dtype=flow.float32,
    device="cuda",
    rng=np.random,
):
    np_x = rng.randn(*shape).astype(np.float32)
    np_weight = rng.randn(*normalized_shape).astype(np.float32) if affine else None

    torch_dtype = torch.float16 if dtype is flow.float16 else torch.float32
    torch_x = torch.tensor(np_x).to(device=device, dtype=torch_dtype)
    torch_weight = (
        torch.tensor(np_weight).to(device=device, dtype=torch_dtype) if affine else None
    )
    torch_x.requires_grad_(True)
    if affine:
        torch_weight.requires_grad_(True)
    torch_y = _torch_rmsnorm(torch_x, torch_weight, normalized_shape, eps)

    np_rand_init_grad = rng.randn(*tuple(torch_y.shape)).astype(np.float32)
    torch_rand_init_grad = torch.tensor(np_rand_init_grad).to(
        device=device, dtype=torch_dtype
    )
    (torch_y * torch_rand_init_grad).sum().backward()

    torch_y = torch_y.detach().cpu().numpy()
    torch_x_grad = torch_x.grad.detach().cpu().numpy()
    if affine:
        torch_weight_grad = torch_weight.grad.detach().cpu().numpy()

    x = flow.tensor(np_x).to(device=device, dtype=dtype)
    weight = flow.tensor(np_weight).to(device=device, dtype=dtype) if affine else None
    x.requires_grad_(True)
    if affine:
        weight.requires_grad_(True)
    y = flow._C.rms_norm(x, weight, normalized_shape, eps)
    # np_rand_init_grad = np.random.randn(*tuple(y.shape)).astype(np.float32)
    rand_init_grad = flow.tensor(np_rand_init_grad).to(device=device, dtype=dtype)
    (y * rand_init_grad).sum().backward()

    y = y.detach().cpu().numpy()
    x_grad = x.grad.detach().cpu().numpy()
    if affine:
        weight_grad = weight.grad.detach().cpu().numpy()

    def compare(a, b, a_name, b_name, atol=1e-5, rtol=1e-8):
        test_case.assertTrue(
            np.allclose(a, b, atol=atol, rtol=rtol),
            f"\n{'=' * 80}"
            f"\n{a_name}:"
            f"\n{a}"
            f"\n{'-' * 80}"
            f"\n{b_name}:"
            f"\n{b}"
            f"\n{'-' * 80}"
            f"\ndiff:"
            f"\n{a - b}"
            f"\n{'*' * 80}"
            f"\nshape={shape}"
            f"\normalized_shape={normalized_shape}"
            f"\naffine={affine}"
            f"\ndtype={dtype}"
            f"\ndevice={device}"
            f"\n{a_name} vs. {b_name} max abs diff: {np.max(np.abs(a - b))}",
        )

    # x_grad subtracts terms of size rstd * |dy| that cancel, so its rounding error grows
    # with each row's rstd. Rows with rms >= 1 keep the plain tolerance.
    norm_dims = _get_norm_dims(shape, normalized_shape)
    mean_square = np.mean(
        np.square(np_x.astype(np.float64)), axis=norm_dims, keepdims=True
    )
    scale = np.maximum(1.0, 1.0 / np.sqrt(mean_square + eps))

    if dtype is flow.float16:
        compare(y, torch_y, "y", "torch_y", 1e-3, 1e-2)
        compare(x_grad, torch_x_grad, "x_grad", "torch_x_grad", 1e-2 * scale, 1e-2)
        if affine:
            compare(
                weight_grad,
                torch_weight_grad,
                "weight_grad",
                "torch_weight_grad",
                0.1,
                0.1,
            )
    else:
        compare(y, torch_y, "y", "torch_y")
        compare(x_grad, torch_x_grad, "x_grad", "torch_x_grad", 1e-5 * scale)
        if affine:
            compare(
                weight_grad,
                torch_weight_grad,
                "weight_grad",
                "torch_weight_grad",
                1e-5,
                1e-4,
            )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestRMSNorm(flow.unittest.TestCase):
    def test_real_example(test_case):
        _test_rmsnorm(
            test_case,
            shape=[512, 4, 768],
            normalized_shape=[768],
            affine=True,
            dtype=flow.float16,
            device="cuda",
        )

    def test_no_affine(test_case):
        _test_rmsnorm(
            test_case, shape=[4, 16], normalized_shape=[16], affine=False,
        )

    def test_widths(test_case):
        for cols in _warp_dispatch_widths():
            _test_rmsnorm(
                test_case,
                shape=[cols % 3 + 2, cols],
                normalized_shape=[cols],
                rng=np.random.RandomState(cols),
            )

    def test_wide_widths(test_case):
        for dtype in [flow.float32, flow.float16]:
            for cols in _wide_widths():
                _test_rmsnorm(
                    test_case,
                    shape=[cols % 3 + 2, cols],
                    normalized_shape=[cols],
                    dtype=dtype,
                    rng=np.random.RandomState(cols),
                )

    def test_warp_impl(test_case):
        _test_rmsnorm(
            test_case, shape=[32, 1024], normalized_shape=[1024], dtype=flow.float16,
        )
        _test_rmsnorm(test_case, shape=[16, 512], normalized_shape=[512])
        _test_rmsnorm(test_case, shape=[15, 512], normalized_shape=[512])
        _test_rmsnorm(test_case, shape=[16, 511], normalized_shape=[511])
        _test_rmsnorm(test_case, shape=[13, 499], normalized_shape=[499])

    def test_block_smem_impl(test_case):
        _test_rmsnorm(
            test_case, shape=[16, 2048], normalized_shape=[2048], dtype=flow.float16,
        )
        _test_rmsnorm(test_case, shape=[8, 1536], normalized_shape=[1536])
        _test_rmsnorm(test_case, shape=[8, 2048], normalized_shape=[2048])
        _test_rmsnorm(test_case, shape=[7, 1536], normalized_shape=[1536])
        _test_rmsnorm(test_case, shape=[8, 1533], normalized_shape=[1533])
        _test_rmsnorm(test_case, shape=[7, 1533], normalized_shape=[1533])

    @unittest.skip("skip for now, becase it failed 4 times in past week")
    def test_block_uncached_impl(test_case):
        _test_rmsnorm(
            test_case,
            shape=[16, 1024 * 1024],
            normalized_shape=[1024 * 1024],
            dtype=flow.float16,
        )
        _test_rmsnorm(
            test_case, shape=[8, 1024], normalized_shape=[1024], dtype=flow.double
        )


if __name__ == "__main__":
    unittest.main()

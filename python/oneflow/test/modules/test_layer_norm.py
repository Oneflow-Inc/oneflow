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

import oneflow as flow
import oneflow.unittest
import torch


def _layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-6):
    begin_norm_axis = len(x.shape) - len(normalized_shape)
    begin_params_axis = len(x.shape) - len(normalized_shape)

    if weight is not None and bias is not None:
        return flow._C.layer_norm_affine(
            x,
            weight,
            bias,
            begin_norm_axis=begin_norm_axis,
            begin_params_axis=begin_params_axis,
            epsilon=eps,
        )
    else:
        return flow._C.layer_norm(
            x,
            begin_norm_axis=begin_norm_axis,
            begin_params_axis=begin_params_axis,
            epsilon=eps,
        )


def _test_layer_norm(
    test_case,
    shape,
    normalized_shape,
    affine=True,
    eps=1e-6,
    dtype=flow.float32,
    device="cuda",
    backward=True,
):
    np_x = np.random.randn(*shape).astype(np.float32)
    if affine:
        np_weight = np.random.randn(*normalized_shape).astype(np.float32)
        np_bias = np.random.randn(*normalized_shape).astype(np.float32)

    # torch process
    torch_dtype = torch.float16 if dtype is flow.float16 else torch.float32
    torch_x = torch.tensor(np_x).to(device=device, dtype=torch_dtype)
    if backward:
        torch_x.requires_grad_(True)
    torch_weight = None
    torch_bias = None
    if affine:
        torch_weight = torch.tensor(np_weight).to(device=device, dtype=torch_dtype)
        torch_bias = torch.tensor(np_bias).to(device=device, dtype=torch_dtype)
        if backward:
            torch_weight.requires_grad_(True)
            torch_bias.requires_grad_(True)
    torch_y = torch.nn.functional.layer_norm(
        torch_x, normalized_shape, torch_weight, torch_bias, eps
    )

    if backward:
        np_rand_init_grad = np.random.randn(*tuple(torch_y.shape)).astype(np.float32)
        torch_rand_init_grad = torch.tensor(np_rand_init_grad).to(
            device=device, dtype=torch_dtype
        )
        (torch_y * torch_rand_init_grad).sum().backward()

        torch_x_grad = torch_x.grad.detach().cpu().numpy()
        if affine:
            torch_weight_grad = torch_weight.grad.detach().cpu().numpy()
            torch_bias_grad = torch_bias.grad.detach().cpu().numpy()

    torch_y = torch_y.detach().cpu().numpy()

    # oneflow process
    x = flow.tensor(np_x).to(device=device, dtype=dtype)
    if backward:
        x.requires_grad_(True)
    weight = None
    bias = None
    if affine:
        weight = flow.tensor(np_weight).to(device=device, dtype=dtype)
        bias = flow.tensor(np_bias).to(device=device, dtype=dtype)
        if backward:
            weight.requires_grad_(True)
            bias.requires_grad_(True)
    y = _layer_norm(x, normalized_shape, weight, bias, eps)

    if backward:
        # np_rand_init_grad = np.random.randn(*tuple(y.shape)).astype(np.float32)
        rand_init_grad = flow.tensor(np_rand_init_grad).to(device=device, dtype=dtype)
        (y * rand_init_grad).sum().backward()

        x_grad = x.grad.detach().cpu().numpy()
        if affine:
            weight_grad = weight.grad.detach().cpu().numpy()
            bias_grad = bias.grad.detach().cpu().numpy()

    y = y.detach().cpu().numpy()

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

    if dtype is flow.float16:
        compare(y, torch_y, "y", "torch_y", 1e-2, 1e-2)
        if backward:
            compare(x_grad, torch_x_grad, "x_grad", "torch_x_grad", 1e-2, 1e-2)
            if affine:
                compare(
                    weight_grad,
                    torch_weight_grad,
                    "weight_grad",
                    "torch_weight_grad",
                    1e-2,
                    1e-2,
                )
                compare(
                    bias_grad,
                    torch_bias_grad,
                    "bias_grad",
                    "torch_bias_grad",
                    1e-2,
                    1e-2,
                )
    else:
        compare(y, torch_y, "y", "torch_y")
        if backward:
            compare(x_grad, torch_x_grad, "x_grad", "torch_x_grad")
            if affine:
                compare(
                    weight_grad, torch_weight_grad, "weight_grad", "torch_weight_grad",
                )
                compare(
                    bias_grad, torch_bias_grad, "bias_grad", "torch_bias_grad",
                )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLayerNorm(flow.unittest.TestCase):
    def test_no_affine(test_case):
        _test_layer_norm(
            test_case, shape=[4, 16], normalized_shape=[16], affine=False,
        )

    def test_warp_impl(test_case):
        _test_layer_norm(
            test_case, shape=[32, 1024], normalized_shape=[1024], dtype=flow.float16,
        )
        _test_layer_norm(test_case, shape=[16, 512], normalized_shape=[512])
        _test_layer_norm(test_case, shape=[15, 512], normalized_shape=[512])
        _test_layer_norm(test_case, shape=[16, 511], normalized_shape=[511])
        _test_layer_norm(test_case, shape=[13, 499], normalized_shape=[499])

    def test_block_smem_impl(test_case):
        _test_layer_norm(
            test_case, shape=[16, 2048], normalized_shape=[2048], dtype=flow.float16,
        )
        _test_layer_norm(test_case, shape=[8, 1536], normalized_shape=[1536])
        _test_layer_norm(test_case, shape=[8, 2048], normalized_shape=[2048])
        _test_layer_norm(test_case, shape=[7, 1536], normalized_shape=[1536])
        _test_layer_norm(test_case, shape=[8, 1533], normalized_shape=[1533])
        _test_layer_norm(test_case, shape=[7, 1533], normalized_shape=[1533])

    def test_block_uncached_impl(test_case):
        _test_layer_norm(
            test_case,
            shape=[16, 1024 * 1024],
            normalized_shape=[1024 * 1024],
            dtype=flow.float16,
        )
        _test_layer_norm(
            test_case, shape=[8, 1024], normalized_shape=[1024], dtype=flow.double
        )


if __name__ == "__main__":
    unittest.main()

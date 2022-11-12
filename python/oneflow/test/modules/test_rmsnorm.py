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

import numpy as np
import unittest
from collections import OrderedDict

import oneflow as flow
import oneflow.unittest
import torch

# from oneflow.test_utils.test_util import GenArgList


def _get_norm_dims(x, weight):
    lpad = len(x.shape) - len(weight.shape)
    assert lpad >= 0
    return tuple(range(lpad, len(x.shape)))


def _torch_rmsnorm(x, weight, eps=1e-6):
    norm_dims = _get_norm_dims(x, weight)
    root_mean = torch.mean(x * x, dim=norm_dims, keepdim=True)
    rms = torch.rsqrt(root_mean + eps)
    normed = x * rms
    return normed * weight if weight is not None else normed


def _test_rmsnorm(
    test_case,
    shape,
    normalized_shape,
    affine=False,
    eps=1e-6,
    dtype=flow.float32,
    device="cuda",
):
    np_x = np.random.randn(*shape).astype(np.float32)
    np_weight = (
        np.random.randn(*normalized_shape).astype(np.float32) if affine else None
    )

    torch_dtype = torch.float16 if dtype is flow.float16 else torch.float32
    # torch_dtype = torch.float32
    torch_x = torch.tensor(np_x).to(device=device, dtype=torch_dtype)
    torch_weight = (
        torch.tensor(np_weight).to(device=device, dtype=torch_dtype) if affine else None
    )
    torch_x.requires_grad_(True)
    torch_weight.requires_grad_(True)
    torch_y = _torch_rmsnorm(torch_x, torch_weight, eps)

    np_rand_init_grad = np.random.randn(*tuple(torch_y.shape)).astype(np.float32)
    torch_rand_init_grad = torch.tensor(np_rand_init_grad).to(
        device=device, dtype=torch_dtype
    )
    (torch_y * torch_rand_init_grad).sum().backward()

    torch_y = torch_y.detach().cpu()
    torch_x_grad = torch_x.grad.detach().cpu()
    torch_weight_grad = torch_weight.grad.detach().cpu()

    x = flow.tensor(np_x).to(device=device, dtype=dtype)
    weight = flow.tensor(np_weight).to(device=device, dtype=dtype) if affine else None
    rand_init_grad = flow.tensor(np_rand_init_grad).to(device=device, dtype=dtype)
    x.requires_grad_(True)
    weight.requires_grad_(True)
    y = flow._C.rms_norm(x, weight, normalized_shape, eps)
    (y * rand_init_grad).sum().backward()

    y = y.detach().cpu()
    x_grad = x.grad.detach().cpu()
    weight_grad = weight.grad.detach().cpu()

    def compare(a, b, a_name, b_name, atol=1e-5, rtol=1e-8):
        test_case.assertTrue(
            np.allclose(a.numpy(), b.numpy(), atol=atol, rtol=rtol),
            f"\n{'=' * 80}"
            f"\n{a_name}:"
            f"\n{a}"
            f"\n{'-' * 80}"
            f"\n{b_name}:"
            f"\n{b}"
            f"\n{'-' * 80}"
            f"\ndiff:"
            f"\n{a.numpy() - b.numpy()}"
            f"\n{'*' * 80}"
            f"\nshape={shape}"
            f"\normalized_shape={normalized_shape}"
            f"\naffine={affine}"
            f"\ndtype={dtype}"
            f"\ndevice={device}"
            f"\n{a_name} vs. {b_name} max abs diff: {np.max(np.abs(a.numpy() - b.numpy()))}",
        )

    if dtype is flow.float16:
        compare(y, torch_y, "y", "torch_y", 1e-3, 1e-2)
        compare(x_grad, torch_x_grad, "x_grad", "torch_x_grad", 1e-2, 1e-2)
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
        compare(x_grad, torch_x_grad, "x_grad", "torch_x_grad")
        if affine:
            compare(
                weight_grad,
                torch_weight_grad,
                "weight_grad",
                "torch_weight_grad",
                1e-5,
                1e-4,
            )


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

    # @autotest(rtol=1e-03, atol=1e-03, check_graph=True)
    # def test_group_norm_with_random_data(test_case):
    #     channels = random(5, 20)
    #     m = torch.nn.GroupNorm(
    #         num_groups=random(1, 5),
    #         num_channels=channels,
    #         eps=random(0, 1) | nothing(),
    #         affine=random(),
    #     )
    #     m.train(random())
    #     device = random_device()
    #     m.to(device)
    #     x = random_tensor(ndim=4, dim1=channels).to(device)
    #     y = m(x)
    #     return y

    # @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    # def test_groupnorm_nhwc(test_case):
    #     _test_groupnorm_nhwc(test_case, (16, 64, 128, 128), 32)


if __name__ == "__main__":
    unittest.main()

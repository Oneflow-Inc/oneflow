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
import oneflow.nn as nn
import oneflow.unittest
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgList

is_profiling = False


def compare_result(test_case, a, b, rtol=1e-5, atol=1e-8):
    test_case.assertTrue(
        np.allclose(a.numpy(), b.numpy(), rtol=rtol, atol=atol),
        f"\na\n{a.numpy()}\n{'-' * 80}\nb:\n{b.numpy()}\n{'*' * 80}\ndiff:\n{a.numpy() - b.numpy()}",
    )


class NaiveSkipRMSNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: flow.Tensor,
        weight: flow.Tensor,
        bias: flow.Tensor = None,
        skip: flow.Tensor = None,
        alpha: float = 1e-5,
        eps: float = 1e-6,
    ) -> flow.Tensor:
        if bias is not None:
            x = flow._C.add(input=x, other=bias)
        if skip is not None:
            skip = skip * alpha
            x = flow._C.add(input=x, other=skip)
        return flow._C.rms_norm(x, weight, [x.shape[-1]], eps)


class FusedSkipRMSNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: flow.Tensor,
        weight: flow.Tensor,
        bias: flow.Tensor = None,
        skip: flow.Tensor = None,
        alpha: float = 1e-5,
        eps: float = 1e-6,
    ) -> flow.Tensor:
        return flow._C.skip_rms_norm(
            x=x, weight=weight, bias=bias, skip=skip, epsilon=eps, alpha=alpha
        )


def _test_skip_rms_norm(
    test_case,
    x_shape,
    has_weight,
    has_bias,
    has_skip,
    eps=1e-6,
    alpha=1e-5,
    dtype=flow.float32,
):
    print(
        f"x_shape: {x_shape}\nhas_weight: {has_weight}\nhas_bias: {has_bias}\nhas_skip: {has_skip}\ndtype: {dtype}\n"
    )

    normalize_shape = list()
    normalize_shape.append(x_shape[-1])

    np_dtype = np.float16 if dtype is flow.float16 else np.float32

    # generate np array
    np_x = np.random.randn(*x_shape).astype(np_dtype)

    naive_flow_weight = None
    fused_flow_weight = None
    if has_weight:
        np_gamma = np.random.randn(*normalize_shape).astype(np_dtype)
        naive_flow_weight = flow.tensor(np_gamma).to(device="cuda", dtype=dtype)
        fused_flow_weight = flow.tensor(np_gamma).to(device="cuda", dtype=dtype)
    else:
        np_gamma = np.ones(*normalize_shape).astype(np_dtype)
        naive_flow_gamma = flow.tensor(np_gamma).to(device="cuda", dtype=dtype)

    flow_bias = None
    if has_bias:
        np_bias = np.random.randn(*normalize_shape).astype(np_dtype)
        flow_bias = flow.tensor(np_bias).to(device="cuda", dtype=dtype)

    flow_skip_naive = None
    flow_skip_fused = None
    np_skip = None
    if has_skip:
        np_skip = np.random.randn(*x_shape).astype(np_dtype)
        flow_skip_naive = flow.tensor(np_skip).to(device="cuda", dtype=dtype)
        flow_skip_fused = flow.tensor(np_skip).to(device="cuda", dtype=dtype)

    # naive process
    flow_naive_module = NaiveSkipRMSNorm()
    flow_x_naive = flow.tensor(np_x).to(device="cuda", dtype=dtype)
    flow_y_naive = flow_naive_module.forward(
        x=flow_x_naive,
        weight=naive_flow_weight,
        bias=flow_bias,
        skip=flow_skip_naive,
        alpha=alpha,
        eps=eps,
    )

    # fused process
    flow_fused_module = FusedSkipRMSNorm()
    flow_x_fused = flow.tensor(np_x).to(device="cuda", dtype=dtype)
    flow_y_fused = flow_fused_module.forward(
        x=flow_x_fused,
        weight=fused_flow_weight,
        bias=flow_bias,
        skip=flow_skip_fused,
        alpha=alpha,
        eps=eps,
    )

    if dtype is flow.float16:
        compare_result(test_case, flow_y_naive, flow_y_fused, 1e-2, 1e-2)
    else:
        compare_result(test_case, flow_y_naive, flow_y_fused, 1e-4, 1e-4)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestSkipRMSNorm(flow.unittest.TestCase):
    def test_gather(test_case):
        arg_dict = OrderedDict()

        # set up test functions
        arg_dict["test_fun"] = [
            _test_skip_rms_norm,
        ]

        # set up test parameters
        if is_profiling:
            arg_dict["x_shape"] = [[1, 5120]]
            arg_dict["has_weight"] = [True]
            arg_dict["has_bias"] = [True]
            arg_dict["has_skip"] = [True]
            arg_dict["eps"] = [1e-6]
            arg_dict["alpha"] = [1e-5]
            arg_dict["dtype"] = [flow.float32]
        else:
            arg_dict["x_shape"] = [[1, 5120]]
            arg_dict["has_weight"] = [True, False]
            arg_dict["has_bias"] = [True, False]
            arg_dict["has_skip"] = [True, False]
            arg_dict["eps"] = [1e-6]
            arg_dict["alpha"] = [1e-5]
            arg_dict["dtype"] = [flow.float32, flow.float16]

        # run test functions
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

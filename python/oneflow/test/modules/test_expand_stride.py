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
import numpy as np


import oneflow as flow
import torch

from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgDict


def _cmp_expand_stride(
    test_case,
    input_shape,
    expand_shape,
    device="cuda",
    disable_view=False,
    verbose=False,
):
    if disable_view:
        origin_env_var = os.environ.get("ONEFLOW_DISABLE_VIEW", "false")
        os.environ["ONEFLOW_DISABLE_VIEW"] = "true"

    input = np.random.randn(*input_shape)
    x = flow.tensor(input, device=device)
    y = x.expand(*expand_shape)

    torch_x = torch.tensor(input, device=device)
    torch_y = torch_x.expand(*expand_shape)

    if verbose:
        print("")
        print(f"{'-' * 10} {input_shape} -> {expand_shape} (disable_view={disable_view}) {'-' * 10}")
        print(f"x.stride={x.stride()}")
        print(input)
        print("*" * 10)
        print(f"y.stride={y.stride()}")
        print(y)
        print("*" * 10)
        print(f"torch_y.stride={torch_y.stride()}")
        print(torch_y)

    test_case.assertTrue(np.array_equal(y.stride(), torch_y.stride()))
    test_case.assertTrue(np.array_equal(y.numpy(), torch_y.detach().cpu().numpy()))

    if disable_view:
        os.environ["ONEFLOW_DISABLE_VIEW"] = origin_env_var


def _cmp_expand_non_contiguous_stride(
    test_case,
    input_shape,
    perm,
    expand_shape,
    device="cuda",
    disable_view=False,
    verbose=False,
):
    if disable_view:
        origin_env_var = os.environ.get("ONEFLOW_DISABLE_VIEW", "false")
        os.environ["ONEFLOW_DISABLE_VIEW"] = "true"

    input = np.random.randn(*input_shape)
    x = flow.tensor(input, device=device)
    y = x.permute(*perm)
    z = y.expand(*expand_shape)

    torch_x = torch.tensor(input, device=device)
    torch_y = torch_x.permute(*perm)
    if disable_view:
        torch_y = torch_y.contiguous()
    torch_z = torch_y.expand(*expand_shape)

    if verbose:
        print("")
        print(
            f"{'-' * 10} {input_shape} with {perm} -> {expand_shape} (disable_view={disable_view}) {'-' * 10}"
        )
        print(f"y: shape={y.shape}, stride={y.stride()}")
        print(f"z: shape={z.shape}, stride={z.stride()}")
        print(f"torch_y: shape={torch_y.shape}, stride={torch_y.stride()}")
        print(f"torch_z: shape={torch_z.shape}, stride={torch_z.stride()}")
        print("*" * 10)
        print(input)
        print("*" * 10)
        print(z)
        print("*" * 10)
        print(torch_z)

    test_case.assertTrue(np.array_equal(z.stride(), torch_z.stride()))
    test_case.assertTrue(np.array_equal(z.numpy(), torch_z.detach().cpu().numpy()))

    if disable_view:
        os.environ["ONEFLOW_DISABLE_VIEW"] = origin_env_var


@flow.unittest.skip_unless_1n1d()
class ExpandStrideTestCase(flow.unittest.TestCase):
    def test_stride(test_case):
        arg_dict = OrderedDict()
        # arg_dict["verbose"] = [False]
        arg_dict["verbose"] = [True]
        arg_dict["device"] = ["cpu", "cuda"]
        # arg_dict["disable_view"] = [True, False]
        arg_dict["disable_view"] = [True]
        arg_dict["shapes"] = [
            # ((2, 2), (2, 2, 2)),
            # ((2, 1, 3), (2, 1, -1, -1, -1)),
            # ((2, 1, 3), (1, 2, -1, -1, -1)),
            # ((2, 1, 3), (2, 1, -1, 2, 3)),
            # ((2, 1, 3), (1, 2, 2, 2, -1)),
            # ((), (3, 2)),
            ((), (2, 1)),
            # ((), (1, 2)),
        ]
        for kwargs in GenArgDict(arg_dict):
            assert "shapes" in kwargs
            input_shape, expand_shape = kwargs.pop("shapes")
            _cmp_expand_stride(test_case, input_shape, expand_shape, **kwargs)

    def test_non_contiguous_stride(test_case):
        arg_dict = OrderedDict()
        arg_dict["verbose"] = [False]
        arg_dict["device"] = ["cpu", "cuda"]
        # when disable_view=True, oneflow will compute stride every op,
        # so flow.permute(flow.randn(2, 1, 3), (0, 2, 1)) will output stride (3, 1, 1),
        # but in pytorch which will output stride (3, 1, 3)
        arg_dict["disable_view"] = [False]
        arg_dict["shapes"] = [
            ((2, 1, 3), (0, 2, 1), (2, 1, -1, -1, -1)),
            ((2, 1, 3), (0, 2, 1), (2, 1, -1, -1, 4)),
            ((2, 1, 3), (0, 2, 1), (3, 2, -1, -1, 4)),
        ]
        for kwargs in GenArgDict(arg_dict):
            assert "shapes" in kwargs
            input_shape, perm, expand_shape = kwargs.pop("shapes")
            _cmp_expand_non_contiguous_stride(
                test_case, input_shape, perm, expand_shape, **kwargs
            )


if __name__ == "__main__":
    unittest.main()

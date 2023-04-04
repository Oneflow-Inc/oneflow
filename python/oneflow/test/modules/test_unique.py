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

import random as random_util
import torch as torch_ori
from collections import OrderedDict

from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

import numpy as np
import oneflow as flow
import oneflow.unittest


def _test_unique_unsorted(test_case, device, return_inverse, return_counts):
    dtype = random_util.choice([torch.int8, torch.int, torch.float, torch.double])
    input = random_tensor(ndim=3, dim0=random(), dim1=random(), dim2=random(), high=20)
    input = input.to(device).to(dtype)
    oneflow_output = flow.unique(
        input.oneflow,
        sorted=False,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    torch_output = torch_ori.unique(
        input.pytorch,
        sorted=False,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    if not return_inverse and not return_counts:
        oneflow_result = oneflow_output
        torch_result = torch_output
    else:
        oneflow_result = oneflow_output[0]
        torch_result = torch_output[0]

    test_case.assertTrue(
        np.allclose(
            np.sort(oneflow_result.numpy()),
            np.sort(torch_result.detach().cpu().numpy()),
        )
    )
    test_case.assertEqual(list(oneflow_result.shape), list(torch_result.shape))

    if return_inverse:
        oneflow_indices = oneflow_output[1]
        torch_indices = torch_output[1]
        test_case.assertTrue(
            np.allclose(
                oneflow_result[oneflow_indices].numpy(),
                torch_result[torch_indices].detach().cpu().numpy(),
            )
        )
        test_case.assertEqual(list(oneflow_indices.shape), list(torch_indices.shape))

    if return_counts:
        oneflow_counts = oneflow_output[-1]
        torch_counts = torch_output[-1]
        test_case.assertTrue(
            np.allclose(
                oneflow_counts.numpy()[np.argsort(oneflow_result.numpy())],
                torch_counts.detach()
                .cpu()
                .numpy()[np.argsort(torch_result.detach().cpu().numpy())],
            )
        )
        test_case.assertEqual(list(oneflow_counts.shape), list(torch_counts.shape))


def _test_unique_sorted(test_case, device, return_inverse, return_counts):
    dtype = random_util.choice([torch.int8, torch.int, torch.float, torch.double])
    input = random_tensor(ndim=3, dim0=random(), dim1=random(), dim2=random(), high=20)
    input = input.to(device).to(dtype)
    oneflow_output = flow.unique(
        input.oneflow,
        sorted=True,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )
    torch_output = torch_ori.unique(
        input.pytorch,
        sorted=True,
        return_inverse=return_inverse,
        return_counts=return_counts,
    )

    if not return_inverse and not return_counts:
        oneflow_result = oneflow_output
        torch_result = torch_output
    else:
        oneflow_result = oneflow_output[0]
        torch_result = torch_output[0]

    test_case.assertTrue(
        np.allclose(oneflow_result.numpy(), torch_result.detach().cpu().numpy(),)
    )
    test_case.assertEqual(list(oneflow_result.shape), list(torch_result.shape))

    if return_inverse:
        oneflow_indices = oneflow_output[1]
        torch_indices = torch_output[1]
        test_case.assertTrue(
            np.allclose(oneflow_indices.numpy(), torch_indices.detach().cpu().numpy(),)
        )
        test_case.assertEqual(list(oneflow_indices.shape), list(torch_indices.shape))

    if return_counts:
        oneflow_counts = oneflow_output[-1]
        torch_counts = torch_output[-1]
        test_case.assertTrue(
            np.allclose(oneflow_counts.numpy(), torch_counts.detach().cpu().numpy(),)
        )
        test_case.assertEqual(list(oneflow_counts.shape), list(torch_counts.shape))


@flow.unittest.skip_unless_1n1d()
class TestUnique(flow.unittest.TestCase):
    @autotest(n=5)
    def test_unique(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["return_inverse"] = [False, True]
        arg_dict["return_counts"] = [False, True]
        for arg in GenArgList(arg_dict):
            _test_unique_unsorted(test_case, *arg)
            _test_unique_sorted(test_case, *arg)

    @profile(torch.unique)
    def profile_unique(test_case):
        input = torch.randint(0, 1000, (1000,))
        torch.unique(input)
        torch.unique(input, return_inverse=True, return_counts=True)
        input = torch.randn(1000,)
        torch.unique(input)
        torch.unique(input, return_inverse=True, return_counts=True)


if __name__ == "__main__":
    unittest.main()

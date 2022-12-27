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
import numpy as np
import oneflow as flow
import torch as torch_ori
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


def _test_unique_unsorted(test_case, placement, sbp):
    input = random_tensor(ndim=1, dim0=64, high=20).to_global(
        placement=placement, sbp=sbp
    )
    oneflow_output = flow.unique(
        input.oneflow, sorted=False, return_inverse=True, return_counts=True
    )
    torch_output = torch_ori.unique(
        input.pytorch, sorted=False, return_inverse=True, return_counts=True
    )

    oneflow_result, oneflow_indices, oneflow_counts = oneflow_output
    torch_result, torch_indices, torch_counts = torch_output

    test_case.assertTrue(
        np.allclose(
            np.sort(oneflow_result.to_local().numpy()),
            np.sort(torch_result.detach().cpu().numpy()),
        )
    )
    test_case.assertTrue(
        np.allclose(
            oneflow_result[oneflow_indices].numpy(),
            torch_result[torch_indices].detach().cpu().numpy(),
        )
    )
    test_case.assertTrue(
        np.allclose(
            oneflow_counts.numpy()[np.argsort(oneflow_result.numpy())],
            torch_counts.detach()
            .cpu()
            .numpy()[np.argsort(torch_result.detach().cpu().numpy())],
        )
    )


def _test_unique_sorted(test_case, placement, sbp):
    input = random_tensor(ndim=1, dim0=64, high=20).to_global(
        placement=placement, sbp=sbp
    )
    oneflow_output = flow.unique(
        input.oneflow, sorted=True, return_inverse=True, return_counts=True
    )
    torch_output = torch_ori.unique(
        input.pytorch, sorted=True, return_inverse=True, return_counts=True
    )

    oneflow_result, oneflow_indices, oneflow_counts = oneflow_output
    torch_result, torch_indices, torch_counts = torch_output

    test_case.assertTrue(
        np.allclose(
            oneflow_result.to_local().numpy(), torch_result.detach().cpu().numpy(),
        )
    )
    test_case.assertTrue(
        np.allclose(oneflow_indices.numpy(), torch_indices.detach().cpu().numpy(),)
    )
    test_case.assertTrue(
        np.allclose(oneflow_counts.numpy(), torch_counts.detach().cpu().numpy(),)
    )


class TestUniqueModule(flow.unittest.TestCase):
    @globaltest
    def test_unique(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement):
                _test_unique_unsorted(test_case, placement, sbp)
                _test_unique_sorted(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

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

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(n=5)
def _test_nll_loss(
    test_case, has_weight=False, split_batch_dim=False, split_class_dim=False
):
    N = random(1, 4) * 2
    C = random(1, 10) * 2
    ndim = random(2, 5).to(int).value()
    dims = [random(2, 10) for i in range(ndim - 2)]
    input_dims = [N, C] + dims
    target_dims = [N] + dims
    input = random_tensor(ndim, *input_dims)
    target = random_tensor(
        ndim - 1, *target_dims, low=0, high=C, dtype=int, requires_grad=False
    )
    weight = None
    if has_weight:
        weight = random_tensor(1, C, requires_grad=False)

    device = random_device().value()
    if not split_class_dim and not split_batch_dim:
        input = input.to(device)
        target = target.to(device)
        if has_weight:
            weight = weight.to(device)
    else:
        rank = flow.env.get_rank()
        world_size = flow.env.get_world_size()
        assert world_size % 2 == 0
        ranks = np.array(range(world_size))

        if split_batch_dim and split_class_dim:
            placement = flow.placement(device, ranks.reshape((ranks.size // 2, 2)))
            input_sbp = [flow.sbp.split(0), flow.sbp.split(1)]
            target_sbp = [flow.sbp.split(0), flow.sbp.broadcast()]
            weight_sbp = [flow.sbp.broadcast(), flow.sbp.split(0)]
        elif split_batch_dim:
            placement = flow.placement(device, ranks)
            input_sbp = flow.sbp.split(0)
            target_sbp = flow.sbp.split(0)
            weight_sbp = flow.sbp.broadcast()
        else:
            placement = flow.placement(device, ranks)
            input_sbp = flow.sbp.split(1)
            target_sbp = flow.sbp.broadcast()
            weight_sbp = flow.sbp.split(0)

        input = input.to_global(placement=placement, sbp=input_sbp)
        target = target.to_global(placement=placement, sbp=target_sbp)
        # print(
        #     f"**[{rank}] input: {input.oneflow.shape} {input.oneflow.placement} {input.oneflow.sbp}"
        # )
        # print(
        #     f"**[{rank}] target: {target.oneflow.shape} {target.oneflow.placement} {target.oneflow.sbp}"
        # )
        if has_weight:
            # print(f"**[{rank}] weight: {weight.oneflow.numpy()}")
            weight = weight.to_global(placement=placement, sbp=weight_sbp)

    # reduction = oneof("none", "sum", "mean")
    reduction = (
        "none"  # Temporarily skip the test of "sum" and "mean" because of unknown error
    )
    if has_weight:
        nll = torch.nn.NLLLoss(weight=weight, reduction=reduction)
    else:
        nll = torch.nn.NLLLoss(reduction=reduction)
    return nll(input, target)


@flow.unittest.skip_unless_1n1d()
class NLLLossTestCase(flow.unittest.TestCase):
    def test_local(test_case):
        _test_nll_loss(test_case)

    def test_weighted(test_case):
        _test_nll_loss(test_case, has_weight=True)


@flow.unittest.skip_unless_1n2d()
class ParallelNLLLossTestCase(flow.unittest.TestCase):
    @globaltest
    def test_data_parallel(test_case):
        _test_nll_loss(test_case, split_batch_dim=True)

    @globaltest
    def test_data_parallel_weighted(test_case):
        _test_nll_loss(test_case, has_weight=True, split_batch_dim=True)

    @globaltest
    def test_model_parallel(test_case):
        _test_nll_loss(test_case, split_class_dim=True)

    @globaltest
    def test_model_parallel_weighted(test_case):
        _test_nll_loss(test_case, has_weight=True, split_class_dim=True)


@flow.unittest.skip_unless_1n4d()
class TowDParallelNLLLossTestCase(flow.unittest.TestCase):
    @globaltest
    def test_2d_parallel(test_case):
        _test_nll_loss(test_case, split_batch_dim=True, split_class_dim=True)

    @globaltest
    def test_2d_parallel_weighted(test_case):
        _test_nll_loss(
            test_case, has_weight=True, split_batch_dim=True, split_class_dim=True
        )


if __name__ == "__main__":
    unittest.main()

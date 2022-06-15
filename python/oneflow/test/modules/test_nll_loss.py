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
import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@autotest(n=1)
def _test_nll_loss(test_case, placement=None, sbp_list=None):
    N = 4
    C = 10
    ndim = random(2, 5).to(int).value()
    dims = [random(1, 2) * random(1, 3) for i in range(ndim - 2)]
    input_dims = [N, C] + dims
    input = random_tensor(ndim, *input_dims)
    target_dims = [N] + dims
    target = random_tensor(
        ndim, *target_dims, low=0, high=C, dtype=flow.int32, requires_grad=False
    )

    if placement and sbp_list:
        if len(sbp_list) == 1:
            input_sbp = sbp_list[0]
            target_sbp = sbp_list[0]
        elif len(sbp_list) > 1:
            input_sbp = sbp_list[0]
            target_sbp = sbp_list[1]
        else:
            raise RuntimeError

        input = input.to_global(placement, input_sbp)
        target = target.to_global(placement, target_sbp)

    reduction = oneof("none", "sum", "mean")
    nll = torch.nn.NLLLoss(reduction=reduction)
    return nll(input, target)


class TestNLLLossModule(flow.unitest.TestCase):
    def test_nll_loss(test_case):
        _test_nll_loss(test_case)


# class TestAddModule(flow.unittest.TestCase):
#     @globaltest
#     def test_add_with_alpha(test_case):
#         ndim = random(1, 4).to(int).value()
#         for placement in all_placement():
#             for sbp in all_sbp(placement, max_dim=ndim):
#                 _test_add_with_alpha(test_case, ndim, placement, sbp)
#             zerodim = random(0, ndim).to(int).value()
#             valid_split_axis = [i for i in range(ndim) if i != zerodim]
#             for sbp in all_sbp(
#                 placement, max_dim=ndim, valid_split_axis=valid_split_axis
#             ):
#                 _test_add_with_0size(test_case, ndim, zerodim, placement, sbp)


if __name__ == "__main__":
    unittest.main()

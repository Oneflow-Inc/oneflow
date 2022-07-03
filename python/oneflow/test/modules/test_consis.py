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

from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest


@autotest(n=1, check_graph=False)
def _test_global_bce_with_logits_reduce_mean_loss_with_random_data(
    test_case, placement, sbp
):
    reduction = "mean"
    m = torch.nn.BCEWithLogitsLoss(weight=None, reduction=reduction, pos_weight=None)
    m.train(random())
    m.to_global(placement=placement, sbp=sbp)
    input = random_tensor(8, 8, 16).to_global(placement, sbp)
    target = random_tensor(8, 8, 16).to_global(placement, sbp)
    y = m(input, target)
    return y


class TestConsistentBCEWithLogitsReduceMeanLoss(flow.unittest.TestCase):
    @globaltest
    def test_global_bce_with_logits_reduce_mean_loss_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=1):
                _test_global_bce_with_logits_reduce_mean_loss_with_random_data(
                    test_case, placement, sbp
                )


if __name__ == "__main__":
    unittest.main()

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


@autotest(n=1, check_graph=True)
def _test_global_triplet_marginloss_with_random_data(test_case, placement, sbp):
    margin = random().to(float)
    p = random().to(float)
    swap = random_bool()
    reduction = oneof("none", "sum", "mean", nothing())
    m = torch.nn.TripletMarginLoss(margin=margin, p=p, swap=swap, reduction=reduction)
    m.train(random())
    anchor = random_tensor(2, 8, 16).to_global(placement, sbp)
    pos = random_tensor(2, 8, 16).to_global(placement, sbp)
    neg = random_tensor(2, 8, 16).to_global(placement, sbp)
    y = m(anchor, pos, neg)
    return y


class TestGlobalTripletMarginLoss(flow.unittest.TestCase):
    @globaltest
    def test_global_triplet_marginloss_with_random_data(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_global_triplet_marginloss_with_random_data(
                    test_case, placement, sbp
                )


if __name__ == "__main__":
    unittest.main()

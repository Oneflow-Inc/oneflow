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
from oneflow.test_utils.test_util import GenArgList

from oneflow.test_utils.automated_test_util import *
import oneflow as flow


@flow.unittest.skip_unless_1n1d()
class TestTripletMarginLoss(flow.unittest.TestCase):
    @autotest(n=10)
    def test_triplet_marginloss_with_random_data(test_case):
        margin = random().to(float)
        p = random().to(float)
        swap = random_bool()
        reduction = oneof("none", "sum", "mean", nothing())
        m = torch.nn.TripletMarginLoss(
            margin=margin, p=p, swap=swap, reduction=reduction
        )
        m.train(random())
        device = random_device()
        m.to(device)
        shape = random_tensor(ndim=2, dim0=random(1, 8)).pytorch.shape
        anchor = random_tensor(len(shape), *shape).to(device)
        pos = random_tensor(len(shape), *shape).to(device)
        neg = random_tensor(len(shape), *shape).to(device)
        y = m(anchor, pos, neg)
        return y


if __name__ == "__main__":
    unittest.main()

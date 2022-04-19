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

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgDict


def _test_coin_flip_impl(test_case, batch_size, random_seed, probability, device):
    m = flow.nn.CoinFlip(batch_size, random_seed, probability, device)
    x = m()
    test_case.assertEqual(x.shape[0], batch_size)
    device = flow.device(device)
    test_case.assertEqual(x.device, device)


class TestCoinFlipModule(flow.unittest.TestCase):
    def test_coin_flip(test_case):
        arg_dict = OrderedDict()
        arg_dict["batch_size"] = [1, 2, 50]
        arg_dict["random_seed"] = [None, 1, -1]
        arg_dict["probability"] = [0.0, 0.5, 1.0]
        # TODO: CoinFlip support cuda kernel
        #  arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["device"] = ["cpu"]

        for arg in GenArgDict(arg_dict):
            _test_coin_flip_impl(test_case, **arg)


if __name__ == "__main__":
    unittest.main()

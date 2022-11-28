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


def _test_global_coin_flip(
    test_case, batch_size, random_seed, probability, placement, sbp
):
    m = flow.nn.CoinFlip(
        batch_size, random_seed, probability, placement=placement, sbp=sbp
    )
    x = m()

    test_case.assertEqual(x.shape[0], batch_size)
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


def _test_graph_coin_flip(
    test_case, batch_size, random_seed, probability, placement, sbp
):
    class GlobalCoinFlipGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()
            self.m = flow.nn.CoinFlip(
                batch_size, random_seed, probability, placement=placement, sbp=sbp
            )

        def build(self):
            return self.m()

    model = GlobalCoinFlipGraph()
    x = model()

    test_case.assertEqual(x.shape[0], batch_size)
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


class TestCoinFlipGlobal(flow.unittest.TestCase):
    @globaltest
    def test_coin_flip_global(test_case):
        arg_dict = OrderedDict()
        arg_dict["batch_size"] = [8, 64]
        arg_dict["random_seed"] = [None, 1, -1]
        arg_dict["probability"] = [0.0, 0.5, 1.0]
        for args in GenArgDict(arg_dict):
            for placement in all_placement():
                # TODO: CoinFlip support cuda kernel
                if placement.type == "cuda":
                    continue

                for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                    _test_global_coin_flip(
                        test_case, **args, placement=placement, sbp=sbp
                    )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n2d()
    def test_coin_flip_graph(test_case):
        arg_dict = OrderedDict()
        arg_dict["batch_size"] = [8]
        arg_dict["random_seed"] = [None, 1, -1]
        arg_dict["probability"] = [0.0, 0.5, 1.0]
        arg_dict["placement"] = [
            # 1d
            flow.placement("cpu", ranks=[0, 1]),
            # TODO: CoinFlip support cuda kernel
            #  flow.placement("cuda", ranks=[0, 1]),
            # 2d
            flow.placement("cpu", ranks=[[0, 1],]),
            # TODO: CoinFlip support cuda kernel
            #  flow.placement("cuda", ranks=[[0, 1],]),
        ]
        for args in GenArgDict(arg_dict):
            placement = args["placement"]
            for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                _test_graph_coin_flip(test_case, **args, sbp=sbp)


if __name__ == "__main__":
    unittest.main()

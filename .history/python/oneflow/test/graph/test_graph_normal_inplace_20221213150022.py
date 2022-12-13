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
import numpy as np
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgDict


_fn_param_local = {
    "normal": lambda data: flow.normal(
        size=data.shape, mean=0.0, std=1.0, out=data
    ),  # NOTE(lixiang): source op that can be inplaced.
}


_fn_param_global = {
    "normal": lambda data, placement, sbp: flow.normal(
        size=data.shape, mean=0.0, std=1.0, out=data, placement=placement, sbp=sbp,
    ),
}


def _test_data_local(test_case, device, fn):

    data_1 = flow.zeros([16, 64, 128, 128]).to(device)
    data_2 = flow.zeros([16, 64, 128, 128]).to(device)

    class NormalGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self):
            fn(data_1).to(device)
            return data_1

    model = NormalGraph()
    lazy_x = model()
    fn(data_2)

    test_case.assertTrue(lazy_x.numpy().sum() != 0)
    test_case.assertTrue(data_2.numpy().sum() != 0)


def _test_data_global(test_case, data_1, data_2, placement, sbp, fn):
    class GlobalNormalGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self):
            flow.manual_seed(233)
            fn(data_1, placement, sbp)
            return data_1

    model = GlobalNormalGraph()
    lazy_x = model()

    flow.manual_seed(233)
    fn(data_2, placement, sbp)

    test_case.assertTrue(
        np.array_equal(lazy_x.to_local().numpy(), data_2.to_local().numpy())
    )


class TestNormalOpInplaceData(flow.unittest.TestCase):
    @oneflow.unittest.skip_unless_1n1d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_normal_op_data_local_with_eager_and_lazy(test_case):

        for device in ["cuda", "cpu"]:
            for _, fn in _fn_param_local.items():
                _test_data_local(test_case, device, fn=fn)

    @globaltest
    def test_normal_op_data_consistent_with_eager_and_lazy(test_case):

        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2, except_partial_sum=True):

                data_1 = flow.empty([8, 64, 128, 128]).to_global(
                    placement=placement, sbp=sbp
                )
                data_2 = flow.empty([8, 64, 128, 128]).to_global(
                    placement=placement, sbp=sbp
                )

                for _, fn in _fn_param_global.items():
                    _test_data_global(test_case, data_1, data_2, placement, sbp, fn=fn)


if __name__ == "__main__":
    unittest.main()

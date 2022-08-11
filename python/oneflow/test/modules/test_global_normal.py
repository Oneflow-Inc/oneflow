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
from oneflow.test_utils.test_util import GenArgList, type_name_to_flow_type
from oneflow.test_utils.automated_test_util import *
import oneflow as flow


def _test_global_normal(
    test_case, placement, sbp, mean, std, shape, dtype, requires_grad
):
    dtype = type_name_to_flow_type[dtype]
    x = flow.normal(
        mean,
        std,
        shape,
        placement=placement,
        sbp=sbp,
        dtype=dtype,
        requires_grad=requires_grad,
    )

    test_case.assertEqual(x.shape, shape)
    test_case.assertEqual(x.dtype, dtype)
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)
    test_case.assertEqual(x.requires_grad, requires_grad)


class TestNormalGlobal(flow.unittest.TestCase):
    @globaltest
    def test_normal_global(test_case):
        arg_dict = OrderedDict()
        arg_dict["mean"] = [-1, 0, 1]
        arg_dict["std"] = [1, 2, 8]
        arg_dict["shape"] = [(8, 8), (8, 8, 8), (8, 8, 8, 8)]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["requires_grad"] = [True, False]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(
                    placement, max_dim=len(arg[2]), except_partial_sum=True
                ):
                    _test_global_normal(test_case, placement, sbp, *arg)


if __name__ == "__main__":
    unittest.main()

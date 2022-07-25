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
from oneflow.test_utils.test_util import GenArgList
import oneflow as flow


def _test_sgd_add_param_group(test_case):
    w1 = flow.ones(3, 3)
    w1.requires_grad = True
    w2 = flow.ones(3, 3)
    w2.requires_grad = True
    o = flow.optim.SGD([w1])
    test_case.assertTrue(o.param_groups[0]["lr"] == 0.001)
    test_case.assertTrue(o.param_groups[0]["momentum"] == 0.0)
    test_case.assertTrue(o.param_groups[0]["weight_decay"] == 0.0)
    test_case.assertTrue(o.param_groups[0]["nesterov"] == False)
    test_case.assertTrue(o.param_groups[0]["maximize"] == False)
    o.step()
    o.add_param_group({"params": w2})
    test_case.assertTrue(o.param_groups[1]["lr"] == 0.001)
    test_case.assertTrue(o.param_groups[1]["momentum"] == 0.0)
    test_case.assertTrue(o.param_groups[1]["weight_decay"] == 0.0)
    test_case.assertTrue(o.param_groups[1]["nesterov"] == False)
    test_case.assertTrue(o.param_groups[1]["maximize"] == False)
    o.step()


class TestAddParamGroup(flow.unittest.TestCase):
    def test_sgd_add_param_group(test_case):
        _test_sgd_add_param_group(test_case)


if __name__ == "__main__":
    unittest.main()

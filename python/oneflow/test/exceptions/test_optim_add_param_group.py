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


@flow.unittest.skip_unless_1n1d()
class TestSgdAddParamGroup(flow.unittest.TestCase):
    def test_sgd_add_param_group_not_unique(test_case):
        with test_case.assertRaises(Exception) as exp:
            w1 = flow.ones(3, 3)
            w1.requires_grad = True
            w2 = flow.ones(3, 3)
            w2.requires_grad = True
            o = flow.optim.SGD([w1])
            o.add_param_group({"params": w2})
            o.add_param_group({"params": w2})
        print(str(exp.exception))
        test_case.assertTrue(
            "some parameters appear in more than one parameter group"
            in str(exp.exception)
        )


if __name__ == "__main__":
    unittest.main()

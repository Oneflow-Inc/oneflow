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
class TestPybind11Caster(flow.unittest.TestCase):
    def test_optional(test_case):
        test_case.assertEqual(
            flow._oneflow_internal.test_api.increase_if_not_none(1), 2
        )
        test_case.assertEqual(
            flow._oneflow_internal.test_api.increase_if_not_none(None), None
        )

    def test_maybe(test_case):
        test_case.assertEqual(flow._oneflow_internal.test_api.divide(6, 2), 3)

    def test_maybe_void(test_case):
        flow._oneflow_internal.test_api.throw_if_zero(1)

    def test_return_maybe_shared_ptr(test_case):
        a1 = flow._oneflow_internal.test_api.get_singleton_a()
        x1 = a1.get_x()
        a1.inc_x()

        a2 = flow._oneflow_internal.test_api.get_singleton_a()
        x2 = a2.get_x()

        test_case.assertEqual(id(a1), id(a2))
        test_case.assertEqual(x1 + 1, x2)

    def test_pass_optional_shared_ptr(test_case):
        a1 = flow._oneflow_internal.test_api.get_singleton_a()
        x1 = a1.get_x()
        a1.inc_x()

        a2 = flow._oneflow_internal.test_api.increase_x_of_a_if_not_none(a1)
        x2 = a2.get_x()

        test_case.assertEqual(id(a1), id(a2))
        test_case.assertEqual(x1 + 2, x2)


if __name__ == "__main__":
    unittest.main()

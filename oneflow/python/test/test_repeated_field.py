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
import oneflow as flow
import oneflow_api.oneflow.core.common.cfg_reflection_test as cfg
import unittest


@flow.unittest.skip_unless_1n1d()
class TestRepeatedField(flow.unittest.TestCase):
    def test_repeated_field(test_case):
        foo = cfg.ReflectionTestFoo()
        bar = cfg.ReflectionTestBar()

        foo.add_repeated_int32(11)
        foo.add_repeated_int32(22)

        foo.add_repeated_string("oneflow")
        foo.add_repeated_string("pytorch")

        bar.mutable_repeated_foo().Add().CopyFrom(foo)

        test_case.assertEqual(
            str(type(foo.repeated_int32())),
            "<class 'oneflow_api.oneflow.core.common.cfg_reflection_test._ConstRepeatedField_<int32_t>'>",
        )
        test_case.assertEqual(
            str(type(foo.repeated_string())),
            "<class 'oneflow_api.oneflow.core.common.cfg_reflection_test._ConstRepeatedField_<::std::string>'>",
        )
        test_case.assertEqual(
            str(type(bar.repeated_foo())),
            "<class 'oneflow_api.oneflow.core.common.cfg_reflection_test._ConstRepeatedField_<::oneflow::cfg::ReflectionTestFoo>'>",
        )


if __name__ == "__main__":
    unittest.main()

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


@flow.unittest.skip_unless_1n1d()
class TestFunctionConfig(flow.unittest.TestCase):
    def test_default_placement_scope(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_placement_scope(flow.scope.placement("cpu", "0:0"))

        @flow.global_function(function_config=func_config)
        def Foo():
            test_case.assertEqual(
                "cpu", flow.current_scope().device_parallel_desc_symbol.device_tag
            )
            return flow.get_variable(
                "w", (10,), initializer=flow.constant_initializer(1)
            )

        Foo().get()

    def test_config_setter_getter(test_case):
        func_config = flow.FunctionConfig()
        func_config.enable_inplace()
        test_case.assertEqual(func_config.function_desc.enable_inplace, True)

    def test_global_function_desc(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_placement_scope(flow.scope.placement("cpu", "0:0"))

        @flow.global_function(function_config=func_config)
        def Foo():
            test_case.assertEqual(
                flow.current_global_function_desc().IsTrainable(), False
            )
            return flow.get_variable(
                "w", (10,), initializer=flow.constant_initializer(1)
            )

        Foo().get()


if __name__ == "__main__":
    unittest.main()

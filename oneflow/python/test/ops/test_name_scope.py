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
import numpy as np
import oneflow as flow


@flow.unittest.skip_unless_1n1d()
class TestNameScope(flow.unittest.TestCase):
    def test_name_scope(test_case):
        flow.clear_default_session()
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)

        def get_var(var_name):
            return flow.get_variable(
                name=var_name,
                shape=(2, 256, 14, 14),
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(),
            )

        @flow.global_function(function_config=func_config)
        def test_name_scope_job():
            with flow.scope.namespace("backbone"):
                with flow.scope.namespace("branch"):
                    var1 = get_var("var")

                with flow.scope.namespace("branch"):
                    var2 = get_var("var")

            var3 = get_var("backbone-branch-var")
            return var1, var2, var3

        var1, var2, var3 = test_name_scope_job().get()
        test_case.assertTrue(np.array_equal(var1.numpy(), var2.numpy()))
        test_case.assertTrue(np.array_equal(var1.numpy(), var3.numpy()))


if __name__ == "__main__":
    unittest.main()

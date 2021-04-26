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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestStatefulLocalKernel(flow.unittest.TestCase):
    def test_stateful_local_kernel(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def job():
            op1 = (
                flow.builtin_op("constant")
                .Output("out")
                .Attr("is_floating_value", True)
                .Attr("floating_value", 3.0)
                .Attr("dtype", flow.float32)
                .Attr("shape", [1, 1])
                .Build()
            )
            op2 = (
                flow.builtin_op("matmul")
                .Input("a")
                .Input("b")
                .Attr("transpose_a", False)
                .Attr("transpose_b", False)
                .Attr("alpha", float(1.0))
                .Output("out")
                .Build()
            )
            x = op1()[0]
            x = op2(x, x)[0]

        job()


if __name__ == "__main__":
    unittest.main()

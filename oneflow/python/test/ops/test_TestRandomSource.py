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


def my_test_source(name, seed):
    return (
        flow.user_op_builder(name)
        .Op("TestRandomSource")
        .Output("out")
        .Attr("seed", seed)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@flow.unittest.skip_unless_1n1d()
class Test_TestRandomSource(flow.unittest.TestCase):
    def test_testsource(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def TestSourceJob():
            with flow.scope.placement("cpu", "0:0"):
                ret = my_test_source("my_cc_test_source_op", 0)
            return ret

        y = TestSourceJob().get().numpy()
        rand_0_4 = np.array([0.5488136, 0.59284467, 0.7151894, 0.8442659, 0.6027634])
        test_case.assertTrue(np.allclose(y, rand_0_4, atol=1e-5, rtol=1e-5))
        y = TestSourceJob().get().numpy()
        if flow.eager_execution_enabled():
            rand_5_9 = rand_0_4
        else:
            rand_5_9 = np.array(
                [0.85794574, 0.54488325, 0.84725183, 0.42365485, 0.62356377]
            )
        test_case.assertTrue(np.allclose(y, rand_5_9, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()

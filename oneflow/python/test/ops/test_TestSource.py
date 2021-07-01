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


def my_test_source(name):
    return (
        flow.user_op_builder(name)
        .Op("TestSource")
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def TODO_test_mirror_testsource(test_case):
    # TODO(chengcheng) source op set mirrored strategy
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.mirrored_view())

    @flow.global_function(function_config=func_config)
    def TestSourceJob():
        with flow.scope.placement("cpu", "0:0"):
            ret = my_test_source("my_cc_test_source_op")
        return ret

    y = TestSourceJob().get().numpy()
    # y = TestSourceJob().get().numpy_list()[0]
    test_case.assertTrue(np.array_equal(y, np.arange(5.0)))


@flow.unittest.skip_unless_1n1d()
class Test_TestSource(flow.unittest.TestCase):
    def test_testsource(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def TestSourceJob():
            with flow.scope.placement("cpu", "0:0"):
                ret = my_test_source("my_cc_test_source_op")
            return ret

        y = TestSourceJob().get().numpy()
        test_case.assertTrue(np.array_equal(y, np.arange(5.0)))


if __name__ == "__main__":
    unittest.main()

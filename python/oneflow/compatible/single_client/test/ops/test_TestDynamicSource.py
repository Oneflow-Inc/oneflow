import os
import unittest

import numpy as np

from oneflow.compatible import single_client as flow


def my_test_source(name):
    return (
        flow.user_op_builder(name)
        .Op("TestDynamicSource")
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@flow.unittest.skip_unless_1n1d()
class Test_TestDynamicSource(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_test_dynamic_source(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def TestSourceJob():
            with flow.scope.placement("cpu", "0:0"):
                ret = my_test_source("my_cc_test_source_op")
            return ret

        y = TestSourceJob().get().numpy_list()[0]
        test_case.assertTrue(np.array_equal(y, np.arange(3.0)))


if __name__ == "__main__":
    unittest.main()

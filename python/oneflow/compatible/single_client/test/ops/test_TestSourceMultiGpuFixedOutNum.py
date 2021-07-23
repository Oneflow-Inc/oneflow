import unittest

import numpy as np

from oneflow.compatible import single_client as flow


def my_test_source(name, out_num):
    return (
        flow.user_op_builder(name)
        .Op("TestSourceMultiGpuFixedOutNum")
        .Output("out")
        .Attr("out_num", out_num)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@flow.unittest.skip_unless_1n1d()
class Test_TestSourceMultiGpuFixedOutNum(flow.unittest.TestCase):
    def test_testsource_2_gpu(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def TestSourceJob():
            with flow.scope.placement("cpu", "0:0-1"):
                ret = my_test_source("my_cc_test_source_op", 10)
            return ret

        y = TestSourceJob().get().numpy()
        test_case.assertTrue(
            np.array_equal(y, np.append(np.arange(5.0), np.arange(5.0)))
        )


if __name__ == "__main__":
    unittest.main()

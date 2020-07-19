import numpy as np
import oneflow as flow


def my_test_source(name):
    return (
        flow.user_op_builder(name)
        .Op("TestDynamicSource")
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def test_test_dynamic_source(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def TestSourceJob():
        with flow.scope.placement("cpu", "0:0"):
            ret = my_test_source("my_cc_test_source_op")
        return ret

    y = TestSourceJob().get().numpy_list()[0]
    test_case.assertTrue(np.array_equal(y, np.arange(3.0)))

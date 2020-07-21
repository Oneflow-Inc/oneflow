import numpy as np
import oneflow as flow


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


def test_testsource_2_gpu(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def TestSourceJob():
        with flow.scope.placement("cpu", "0:0-1"):
            ret = my_test_source("my_cc_test_source_op", 10)
        # print("cons_test_source_batch_axis", ret.batch_axis)
        test_case.assertTrue(ret.batch_axis is not None and ret.batch_axis == 0)
        return ret

    y = TestSourceJob().get().numpy()
    test_case.assertTrue(np.array_equal(y, np.append(np.arange(5.0), np.arange(5.0))))

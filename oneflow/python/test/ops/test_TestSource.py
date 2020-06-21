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


def test_testsource(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.global_function(func_config)
    def TestSourceJob():
        with flow.fixed_placement("cpu", "0:0"):
            ret = my_test_source("my_cc_test_source_op")
        # print("cons_test_source_batch_axis", ret.batch_axis)
        test_case.assertTrue(ret.batch_axis is not None and ret.batch_axis == 0)
        return ret

    y = TestSourceJob().get().ndarray()
    test_case.assertTrue(np.array_equal(y, np.arange(5.0)))


def TODO_test_mirror_testsource(test_case):
    # TODO(chengcheng) source op set mirrored strategy
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.global_function(func_config)
    def TestSourceJob():
        with flow.device_prior_placement("cpu", "0:0"):
            ret = my_test_source("my_cc_test_source_op")
        # print("mirr_test_source_batch_axis", ret.batch_axis)
        test_case.assertTrue(ret.batch_axis is not None and ret.batch_axis == 0)
        return ret

    y = TestSourceJob().get().ndarray()
    # y = TestSourceJob().get().ndarray_list()[0]
    test_case.assertTrue(np.array_equal(y, np.arange(5.0)))

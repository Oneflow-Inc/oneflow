import numpy as np
import oneflow as flow


def my_test_source(name, seed):
    return (
        flow.user_op_builder(name)
        .Op("TestRandomSource")
        .Output("out")
        .Attr("seed", seed, "AttrTypeInt64")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def test_testsource(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.scope.consistent_view())

    @flow.global_function(func_config)
    def TestSourceJob():
        with flow.fixed_placement("cpu", "0:0"):
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

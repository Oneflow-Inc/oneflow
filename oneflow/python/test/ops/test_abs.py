import oneflow as flow
import numpy as np

def test_abs(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def AbsJob(a=flow.FixedTensorDef((5, 2))):
        return flow.math.abs(a)

    x = np.random.rand(5, 2).astype(np.float32)
    y = AbsJob(x).get().ndarray()
    test_case.assertTrue(np.array_equal(y, np.absolute(x)))

def test_mirror_abs(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def AbsJob(a=flow.FixedTensorDef((5, 2))):
        return flow.user_op_builder("abs_op_name").Op("unary")\
                .Input("x",[a])\
                .Output("y")\
                .SetAttr("unary_math_type", "Abs", "AttrTypeString")\
                .Build().RemoteBlobList()[0]

    x = np.random.rand(5, 2).astype(np.float32)
    y = AbsJob(x).get().ndarray()
    test_case.assertTrue(np.array_equal(y, np.absolute(x)))


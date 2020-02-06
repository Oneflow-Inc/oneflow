import oneflow as flow
import numpy as np

def ccrelu(x, name):
    return flow.user_op_builder(name).Op("ccrelu").Input("in",[x]).Output("out").Build().RemoteBlobList()[0]

def test_ccrelu(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.function(func_config)
    def ReluJob(a=flow.FixedTensorDef((5, 2))):
        return ccrelu(a, "my_cc_relu_op")

    x = np.random.rand(5, 2).astype(np.float32)
    y = ReluJob(x).get().ndarray()
    test_case.assertTrue(np.array_equal(y, np.maximum(x, 0)))

def test_mirror_ccrelu(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def ReluJob(a = flow.MirroredTensorDef((5, 2))):
        return ccrelu(a, "my_cc_relu_op")

    x = np.random.rand(5, 2).astype(np.float32)
    y = ReluJob([x]).get().ndarray_list()[0]
    test_case.assertTrue(np.array_equal(y, np.maximum(x, 0)))


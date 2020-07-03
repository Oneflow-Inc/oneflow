import numpy as np
import oneflow as flow


def ccrelu(x, name):
    return (
        flow.user_op_builder(name)
        .Op("ccrelu")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def fixed_tensor_def_test(test_case, func_config):
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def ReluJob(a=flow.FixedTensorDef((5, 2))):
        return ccrelu(a, "my_cc_relu_op")

    x = np.random.rand(5, 2).astype(np.float32)
    y = ReluJob(x).get().ndarray()
    test_case.assertTrue(np.array_equal(y, np.maximum(x, 0)))


def mirrored_tensor_def_test(test_case, func_config):
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def ReluJob(a=flow.MirroredTensorDef((5, 2))):
        return ccrelu(a, "my_cc_relu_op")

    x = np.random.rand(3, 1).astype(np.float32)
    y = ReluJob([x]).get().ndarray_list()[0]
    test_case.assertTrue(np.array_equal(y, np.maximum(x, 0)))


def test_ccrelu(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    fixed_tensor_def_test(test_case, func_config)


def test_mirror_ccrelu(test_case):
    func_config = flow.FunctionConfig()
    mirrored_tensor_def_test(test_case, func_config)


def test_1n2c_mirror_dynamic_ccrelu(test_case):
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(func_config)
    def ReluJob(a=flow.MirroredTensorDef((5, 2))):
        return ccrelu(a, "my_cc_relu_op")

    x1 = np.random.rand(3, 1).astype(np.float32)
    x2 = np.random.rand(4, 2).astype(np.float32)
    y1, y2 = ReluJob([x1, x2]).get().ndarray_list()
    test_case.assertTrue(np.array_equal(y1, np.maximum(x1, 0)))
    test_case.assertTrue(np.array_equal(y2, np.maximum(x2, 0)))


@flow.unittest.num_nodes_required(2)
def test_ccrelu_2n1c(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    fixed_tensor_def_test(test_case, func_config)

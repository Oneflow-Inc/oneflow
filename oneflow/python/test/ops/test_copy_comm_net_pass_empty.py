import oneflow as flow
import numpy as np


def ccrelu(x, name):
    return flow.user_op_builder(name)\
            .Op("ccrelu")\
            .Input("in",[x])\
            .Output("out")\
            .Build().RemoteBlobList()[0]

@flow.unittest.num_nodes_required(2)
def test_multi_node_comm_net(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_data_type(flow.float)
    flow.config.gpu_device_num(1)

    @flow.function(func_config)
    def ReluJob(x = flow.FixedTensorDef((10, 2))):
        with flow.fixed_placement("gpu", "0:0"):
            out0 = ccrelu(x, "my_op_0_0")
        with flow.fixed_placement("gpu", "1:0"):
            out1 = ccrelu(out0, "my_op_1_0")
        with flow.fixed_placement("gpu", "0:0"):
            out2 = ccrelu(out1, "my_op_print")
        return out2
    index = [-2, -1, 0, 1, 2]
    data = []
    for i in index: data.append(np.ones((10, 2,), dtype=np.float32) * i)
    for i in range(5):
        ret = ReluJob(data[i]).get().ndarray()
        print(ret)
        if index[i] > 0 :
            test_case.assertTrue(np.array_equal(ret, np.ones((10, 2,), dtype=np.float32) * index[i]))
        else:
            test_case.assertTrue(np.array_equal(ret, np.zeros((10, 2,), dtype=np.float32)))

@flow.unittest.num_nodes_required(2)
def test_multi_node_comm_net_dynamic(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())
    func_config.default_placement_scope(flow.fixed_placement("gpu", "0:0"))
    func_config.default_data_type(flow.float)
    flow.config.machine_num(2)
    flow.config.gpu_device_num(1)

    @flow.function(func_config)
    def ReluJob(x = flow.MirroredTensorDef((10, 2))):
        with flow.fixed_placement("gpu", "0:0"):
            out0 = flow.keras.activations.relu(x)
        with flow.fixed_placement("gpu", "1:0"):
            out1 = flow.keras.activations.relu(out0)
        with flow.fixed_placement("gpu", "0:0"):
            out2 = flow.keras.activations.relu(out1)
        return out2
    index = [-2, -1, 0, 1, 2]
    data = []
    for i in index: data.append(np.ones((5, 2,), dtype=np.float32) * i)
    for i in range(5):
        ret = ReluJob([data[i]]).get().ndarray_list()[0]
        print(ret)
        if index[i] > 0 :
            test_case.assertTrue(np.array_equal(ret, np.ones((5, 2,), dtype=np.float32) * index[i]))
        else:
            test_case.assertTrue(np.array_equal(ret, np.zeros((5, 2,), dtype=np.float32)))

@flow.unittest.num_nodes_required(2)
def test_multi_node_comm_net_dynamic_empty(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    func_config.default_data_type(flow.float)
    flow.config.machine_num(2)
    flow.config.gpu_device_num(1)

    @flow.function(func_config)
    def ReluJob(x = flow.MirroredTensorDef((10, 2))):
        with flow.fixed_placement("cpu", "0:0"):
            out0 = flow.keras.activations.relu(x)
        with flow.fixed_placement("cpu", "1:0"):
            out1 = flow.keras.activations.relu(out0)
        with flow.fixed_placement("cpu", "0:0"):
            out2 = flow.keras.activations.relu(out1)
        return out2
    index = [-2, -1, 0, 1, 2]
    data = []
    for i in index: data.append(np.ones((0,0,), dtype=np.float32) * i)
    for i in range(5):
        ret = ReluJob([data[i]]).get().ndarray_list()[0]
        print(ret)
        if index[i] > 0 :
            test_case.assertTrue(np.array_equal(ret, np.ones((0, 0,), dtype=np.float32) * index[i]))
        else:
            test_case.assertTrue(np.array_equal(ret, np.zeros((0, 0,), dtype=np.float32)))


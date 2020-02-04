import oneflow as flow
import numpy as np

@flow.unittest.num_nodes_required(2)
def test_multi_node_dynamic_binary_split_concat_empty(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    func_config.default_data_type(flow.float)
    flow.config.machine_num(2)
    flow.config.gpu_device_num(1)

    @flow.function(func_config)
    def DynamicBinaryJob(x = flow.MirroredTensorDef((20,))):
        print("in_shape: ", x.shape)
        with flow.fixed_placement("cpu", "0:0"):
            out_list = flow.experimental.dynamic_binary_split(x, base_shift=4, out_num=6)
            id_out_list = []
            for out_blob in out_list:
                print("out_shape: ", out_blob.shape)
                id_out_list.append(flow.identity(out_blob))
        with flow.fixed_placement("cpu", "1:0"):
            out1 = flow.experimental.dynamic_binary_concat(id_out_list, x)
            print("concat_shape: ", out1.shape)
        with flow.fixed_placement("cpu", "0:0"):
            out2 = flow.identity(out1)
            print("return_shape: ", out2.shape)
        return out2
    size = [0, 5, 10, 15, 20]
    data = []
    for i in size: data.append(np.ones((i,), dtype=np.float32))
    for i in range(5):
        ret = DynamicBinaryJob([data[i]]).get().ndarray_list()[0]
        print(ret)
        test_case.assertTrue(np.array_equal(ret, data[i]))


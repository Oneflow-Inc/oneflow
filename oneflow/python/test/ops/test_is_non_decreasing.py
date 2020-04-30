import oneflow as flow
import numpy as np

def test_is_non_decreasing(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    func_config.default_data_type(flow.float)

    arr_length = 5
    @flow.function(func_config)
    def IsNDJob(x = flow.FixedTensorDef((arr_length,), batch_axis=None, dtype=flow.int32)):
        return flow.math.is_non_decreasing(x)

    data = []
    data.append(np.array([1,2,3,4,5],dtype=np.int32))
    data.append(np.array([1,1,2,2,3],dtype=np.int32))
    data.append(np.array([1,2,3,2,1],dtype=np.int32))

    y_groundtruth=[]
    y_groundtruth.append(np.array([1],dtype=np.int8))
    y_groundtruth.append(np.array([1],dtype=np.int8))
    y_groundtruth.append(np.array([0],dtype=np.int8))

    for i in range(len(data)):
        y=IsNDJob(data[i]).get().ndarray()
        test_case.assertTrue(np.array_equal(y,y_groundtruth[i]))
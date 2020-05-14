import oneflow as flow
import numpy as np

def test_invert_permutation(test_case):
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def InvertJob(x = flow.FixedTensorDef((5,),dtype=flow.int32)):
        return flow.math.invert_permutation(x, "xx_i_p")

    data = []

    data.append(np.array([4,3,2,1,0],dtype=np.int32))
    data.append(np.array([0,1,2,3,4],dtype=np.int32))
    data.append(np.array([3,4,0,2,1],dtype=np.int32))

    y_groundtruth=[]
    y_groundtruth.append(np.array([4,3,2,1,0],dtype=np.int32))
    y_groundtruth.append(np.array([0,1,2,3,4],dtype=np.int32))
    y_groundtruth.append(np.array([2,4,3,0,1],dtype=np.int32))    
    for i in range(len(data)): 
        y=InvertJob(data[i]).get().ndarray()
        test_case.assertTrue(np.allclose(y,y_groundtruth[i]))
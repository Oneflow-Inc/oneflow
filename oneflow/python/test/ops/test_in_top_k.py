import oneflow as flow
import numpy as np

def compare(prediction,target,k,y_groundtruth):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
    func_config.default_placement_scope(flow.fixed_placement("cpu", "0:0"))
    func_config.default_data_type(flow.float)


    def test_in_top_k(predictions, targets, k, name=None):
        if name is None:
            name = id_util.UniqueStr("intopk_")
        return flow.user_op_builder(name).Op("in_top_k")\
                .Input("predictions",[predictions])\
                .Input("targets",[targets])\
                .SetAttr("k", k, "AttrTypeInt32",)\
                .Output("out")\
                .Build().RemoteBlobList()[0]


    @flow.function(func_config)
    def IntopkJob(predictions=flow.FixedTensorDef((3,4),dtype=flow.float),targets=flow.FixedTensorDef((3,),dtype=flow.int32)):
        return test_in_top_k(predictions, targets , k ,"xx_in_top_k")

    y=IntopkJob(prediction,target).get().ndarray()
    assert (np.allclose(y,y_groundtruth))

def test(test_case):

    data = []

    data.append(np.array([[ 0.46714601 ,0.92652822 ,0.16808732 , 0.44906664],
 [ 0.03874864 , 0.55331773 , 0.32944077 , 0.84536946],
 [ 0.80283058 , 0.63945484 , 0.07212774 , 0.27699497]],dtype=np.float32))
    data.append(np.array([[ 0.46714601 ,0.92652822 ,0.16808732 , 0.44906664],
 [ 0.03874864 , 0.55331773 , 0.32944077 , 0.84536946],
 [ 0.80283058 , 0.63945484 , 0.07212774 , 0.27699497]],dtype=np.float32))

    target=[]
    target.append(np.array([3,3,3],dtype=np.int32))
    target.append(np.array([3,3,3],dtype=np.int32))

    k=[1,3]

    y_groundtruth=[]
    y_groundtruth.append(np.array([0,1,0]))
    y_groundtruth.append(np.array([1,1,1]))

    for i in range(len(data)): 
        compare(data[i],target[i],k[i],y_groundtruth[i])

import oneflow as flow
import numpy as np
print('hello')
#flow.config.gpu_device_num(1)

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
print('hello')
def test_i_p(x, name):
    return flow.user_op_builder(name).Op("invert_permutation").Input("in",[x]).Output("out") \
            .Build().RemoteBlobList()[0]

@flow.function(func_config)
def ReshapeJob(x = flow.FixedTensorDef((3,))):
    return test_i_p(x, "xx_i_p")


data = []
print('hello')
index=[1,2,3]
for i in index: data.append(np.array([0,2,1]))
for x in data: print(ReshapeJob(x).get())

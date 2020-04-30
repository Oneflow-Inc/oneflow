import numpy as np
import oneflow as flow

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
@flow.function(func_config)
def IsSIJob(input = flow.FixedTensorDef((5,2),  dtype=flow.float)):
    z = flow.math.reduce_all(input,axis=[0,1])
    return z

input = np.random.rand(5,2).astype(np.float32)
print(input)
y = IsSIJob(input)
print(y.get())
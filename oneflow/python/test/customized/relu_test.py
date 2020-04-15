import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)
#func_config.train.primary_lr(1e-4)
#func_config.train.model_update_conf(dict(naive_conf={}))

shape = (4, 3)
@flow.function(func_config)
def ReluJob(x = flow.FixedTensorDef(shape, dtype=flow.float)):
    with flow.device_prior_placement('gpu', "0:0"):
        return flow.nn.relu(x)

x = np.random.rand(*shape).astype(np.float32) - 0.5
print(x)
of_out = ReluJob(x).get()
print(of_out.ndarray())
import numpy as np
import oneflow as flow

flow.config.gpu_device_num(1)

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)


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


@flow.global_function(func_config)
def ReluJob(x=flow.FixedTensorDef((10, 2))):
    return ccrelu(x, "my_cc_relu_op")


index = [-2, -1, 0, 1, 2]
data = []
for i in index:
    data.append(np.ones((10, 2,), dtype=np.float32) * i)
for x in data:
    print(ReluJob(x).get())

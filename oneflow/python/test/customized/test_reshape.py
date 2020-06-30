import numpy as np
import oneflow as flow

flow.config.gpu_device_num(1)

func_config = flow.FunctionConfig()
func_config.default_distribute_strategy(flow.distribute.consistent_strategy())
func_config.default_data_type(flow.float)


def test_reshape(x, shape, name):
    return (
        flow.user_op_builder(name)
        .Op("TestReshape")
        .Input("in", [x])
        .Output("out")
        .Attr("shape", shape, "AttrTypeShape")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@flow.global_function(func_config)
def ReshapeJob(x=flow.FixedTensorDef((10, 2))):
    return test_reshape(x, [5, 4], "xx_test_reshape")


index = [2.22, -1, 0, 1.1, 2]
data = []
for i in index:
    data.append(np.ones((10, 2,), dtype=np.float32) * i)
for x in data:
    print(ReshapeJob(x).get())

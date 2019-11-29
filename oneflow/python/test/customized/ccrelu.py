import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

def ccrelu(x, name):
    return flow.user_op_builder(name).Op("ccrelu").Input("in",[x]).Build().RemoteBlobList()[0]

@flow.function
def ReluJob(x = flow.input_blob_def((10, 2))):
    return ccrelu(x, "my_cc_relu_op")

index = [-2, -1, 0, 1, 2]
data = []
for i in index: data.append(np.ones((10, 2,), dtype=np.float32) * i)
for x in data:  print(ReluJob(x).get())

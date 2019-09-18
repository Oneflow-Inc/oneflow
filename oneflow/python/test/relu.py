import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

@flow.function
def ReluJob(x = flow.input_blob_def((10,))):
    return flow.keras.activations.relu(x)

index = [-2, -1, 0, 1, 2]
data = []
for i in index: data.append(np.ones((10,), dtype=np.float32) * i)
for x in data:  print(ReluJob(x).get())

import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.grpc_use_no_signal()

flow.config.piece_size(10)
flow.config.default_data_type(flow.float)

def Print(x, y):
    print("x: ")
    print(x)
    print("y: ")
    print(y)

@flow.function
def ReluJob(x = flow.input_blob_def((10,))):
    y = flow.keras.activations.relu(x)
    flow.watch([x, y], Print)

index = [-2, -1, 0, 1, 2]
data = []
for i in index: data.append(np.ones((10,), dtype=np.float32) * i)
for x in data:  ReluJob(x)

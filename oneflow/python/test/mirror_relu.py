import oneflow as flow
import numpy as np

flow.config.gpu_device_num(2)
flow.config.default_data_type(flow.float)

@flow.function
def ReluJob(x = flow.mirror_blob_def((10,))):
    return flow.keras.activations.relu(flow.keras.activations.sigmoid(x))

x = np.ones((10,), dtype=np.float32)
print(ReluJob((x, x)).get())

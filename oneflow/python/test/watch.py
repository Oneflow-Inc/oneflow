import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

@flow.function
def ReluJob(x = flow.input_blob_def((10,))):
    flow.config.train.primary_lr(0.5)
    flow.config.train.model_update_conf(dict(naive_conf={}))
    w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(10))
    y = flow.keras.activations.relu(x) + w
    flow.losses.add_loss(y)
    flow.watch(y, "y:")
    return y

index = [-2, -1, 0, 1, 2]
data = []
for i in index: data.append(np.ones((10,), dtype=np.float32) * i)
for x in data:  ReluJob(x)

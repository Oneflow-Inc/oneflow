import oneflow as flow
import numpy as np
import torch
import math

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

@flow.function
def SigmoidJob(x = flow.input_blob_def((10,))):
    return flow.keras.activations.sigmoid(x)

x = np.array(range(-5,5), dtype=np.float32)

a = SigmoidJob(x).get()

b = torch.sigmoid(torch.Tensor(x))

result = np.isclose(np.array(a), b.numpy(), rtol=1e-03, atol=1e-05)
for i in result:
    assert i, "the sigmoid test is wrong!"

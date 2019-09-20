import oneflow as flow
import numpy as np
import torch
import math

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

@flow.function
def GeluJob(x = flow.input_blob_def((10,))):
    return flow.keras.activations.gelu(x)

x = np.ones((10,), dtype=np.float32)

a = GeluJob(x).get()

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * torch.pow(x, 3))))
x = torch.tensor(x)
b = gelu(x)

result = np.isclose(np.array(a), b.numpy(), rtol=1e-03, atol=1e-05)
for i in result:
    assert i, "the test is wrong!"

import oneflow as flow
import numpy as np
import numpy as np
import math

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)
flow.config.ctrl_port(2334)

@flow.function
def GeluJob(x = flow.input_blob_def((10,))):
    return flow.keras.activations.gelu(x)

x = np.random.rand(10,).astype(np.float32)
print("in:", x)

a = GeluJob(x).get()
print("of_out:", a)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x, 3))))
b = gelu(x)
print("np.out:", a)

result = np.isclose(np.array(a), b, rtol=1e-03, atol=1e-05)
print(result)
for i in result:
    assert i, "the test is wrong!"

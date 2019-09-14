import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.grpc_use_no_signal()

@flow.function
def TestNet(x=flow.input_blob_def((1,))):
    return (x, x)

x = np.array([1], dtype=np.float32)
fetched = TestNet(x).get()
print(fetched)

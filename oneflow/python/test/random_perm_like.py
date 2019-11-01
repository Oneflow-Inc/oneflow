import oneflow as flow
import numpy as np
import time

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

SIZE = 10
@flow.function
def TestJob(x = flow.input_blob_def((SIZE,), dtype=flow.float32, is_dynamic=True)):
    return flow.argsort(x)
    return flow.detection.random_perm_like(x) 

for i in range(4):
    r = TestJob(np.zeros(SIZE).astype(np.float32)).get()
    print(r)

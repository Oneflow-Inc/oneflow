import oneflow as flow
import numpy as np
import time

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

SHAPE = (100000,)
@flow.function
def TestJob(x = flow.input_blob_def(SHAPE, dtype=flow.float32, is_dynamic=True)):
    return flow.detection.random_perm_like(x + x) 

for i in range(4):
    r = TestJob(np.zeros(SHAPE).astype(np.float32)).get()
    print(r)

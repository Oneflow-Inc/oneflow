import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

SIZE = 10
@flow.function
def TestJob(x = flow.input_blob_def((SIZE,), dtype=flow.float32, is_dynamic=True)):
    
    return flow.detection.random_perm_like(x + x) 

print(TestJob(np.random.randn(SIZE).astype(np.float32)).get())
print(TestJob(np.random.randn(SIZE).astype(np.float32)).get())
print(TestJob(np.random.randn(SIZE).astype(np.float32)).get())
print(TestJob(np.random.randn(SIZE).astype(np.float32)).get())

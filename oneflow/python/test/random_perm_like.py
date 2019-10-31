import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

@flow.function
def TestJob(x = flow.input_blob_def((4,), dtype=flow.float32)):
    
    return flow.detection.random_perm_like(x + x) 

print(TestJob(np.random.randn(4).astype(np.float32)).get())
print(TestJob(np.random.randn(4).astype(np.float32)).get())
print(TestJob(np.random.randn(4).astype(np.float32)).get())
print(TestJob(np.random.randn(4).astype(np.float32)).get())

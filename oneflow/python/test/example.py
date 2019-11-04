import oneflow as flow
import numpy as np

config.gpu_device_num(1)

@flow.function
def DemoJob(x = flow.input_blob_def((10,))):
    return x

data = []
for i in range(5): data.append(np.ones((10,), dtype=np.float32) * i)
print "DemoJob(x).get()"
for x in data: print DemoJob(x).get()
def PrintRunAsyncResult(x):
    print x
print "DemoJob(x).async_get(...)"
for x in data: DemoJob(x).async_get(PrintRunAsyncResult)

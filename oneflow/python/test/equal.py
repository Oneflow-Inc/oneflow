import oneflow as flow
import numpy as np

@flow.function
def EqualJob(a=flow.input_blob_def((5, 2)), b=flow.input_blob_def((5, 2))):
    flow.config.default_data_type(flow.float)
    return a == b


x = np.random.rand(5, 2).astype(np.float32)
y = np.random.rand(5, 2).astype(np.float32)
z = None

z = EqualJob(x, y).get()
print (z)
z = EqualJob(x, x).get()
print (z)

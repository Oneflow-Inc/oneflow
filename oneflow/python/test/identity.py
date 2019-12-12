import oneflow as flow
import numpy as np


@flow.function
def IdJob(a=flow.input_blob_def((1, 4,))):
    with flow.device_prior_placement('cpu', '0:0'):
      flow.config.default_data_type(flow.float)
      y = flow.identity(a)
      return y

x = np.random.rand(1, 4).astype(np.float32)
y = IdJob(x).get().ndarray()
print(x)
print("----")
print(y)
print(np.array_equal(x, y))

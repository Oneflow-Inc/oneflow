import oneflow as flow
import numpy as np

flow.env.grpc_use_no_signal()
flow.config.gpu_device_num(2)

flow.config.default_data_type(flow.float)

def Print(prefix):
    def _print(x):
        print(prefix)
        print(x)
    return _print

@flow.function
def DistributeConcat(
      a = flow.input_blob_def((2, 5), is_dynamic=False),
      b = flow.input_blob_def((2, 5), is_dynamic=False)):
  with flow.device_prior_placement("gpu", "0:0"):
    a = flow.math.relu(a)
    b = flow.math.relu(b)
  ret = flow.debug.distribute_concat([a, b])
  print(ret.shape)

index = [-1, 0, 1, 2, 3]
data = []
for i in index: data.append(np.ones((2, 5), dtype=np.float32) * i)
for x in data:
  DistributeConcat(x, x)

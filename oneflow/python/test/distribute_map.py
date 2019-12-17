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
def DistributeMap(a = flow.input_blob_def((2, 5), is_dynamic=False)):
  a = flow.math.relu(a)
  ret = flow.advanced.distribute_map(a, flow.math.relu)
  print(ret.shape)
  return ret

index = [-1, 0, 1, 2, 3]
data = []
for i in index: data.append(np.ones((2, 5), dtype=np.float32) * i)
for x in data:
  DistributeMap(x)

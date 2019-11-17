import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.grpc_use_no_signal()

flow.config.default_data_type(flow.float)

def Print(prefix):
    def _print(x):
        print(prefix)
        print(x)
    return _print

blob_watched = {}

@flow.function
def ReluJob(x = flow.input_blob_def((12, 1), is_dynamic=True)):
    print(x.is_dynamic)
    with flow.watch_scope(blob_watched):
      y = flow.keras.activations.relu(x)
      z = flow.keras.activations.relu(y)
      print(y.is_dynamic)
      return z

index = [-2, -1, 0, 1, 2]
data = []
for i in index: data.append(np.ones((2, 5), dtype=np.float32) * i)
for x in data:
  ReluJob(x).get()
  for lbn, blob_data in blob_watched.items():
    print(lbn)
    print(blob_data['blob_def'].location)
    print(blob_data['blob'])

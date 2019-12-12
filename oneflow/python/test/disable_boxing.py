import oneflow as flow
import numpy as np

flow.config.gpu_device_num(2)
flow.config.grpc_use_no_signal()

flow.config.default_data_type(flow.float)

def Print(prefix):
    def _print(x):
        print(prefix)
        print(x)
    return _print

blob_watched = {}

@flow.function
def ReluJob(x = flow.input_blob_def((2, 5))):
    print("x.disable_boxing: " , x.disable_boxing)
    with flow.device_prior_placement("gpu", "0:0"):
      y = flow.math.relu(x).with_boxing_disabled()
    w = flow.get_variable("w", shape=(1, 5), initializer=flow.constant_initializer(0))
    z = y + w
    print("w.disable_boxing: " , w.disable_boxing)
    print("w.parallel_conf: " , w.parallel_conf)
    print("z.disable_boxing: " , z.disable_boxing)
    print("z.parallel_conf: " , z.parallel_conf)
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

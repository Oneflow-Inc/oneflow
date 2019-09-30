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

@flow.function
def IdJob(x = flow.input_blob_def((2, 5, 5), num_of_lod_levels=2)):
    return x

index = [-2, -1, 0, 1, 2]
data = []
for i in index: data.append(np.ones((2, 5), dtype=np.float32) * i)
for x in data:  print(IdJob([x, x]).get().lod_ndarray_nested_list())

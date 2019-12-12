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
def TestJob(x1 = flow.input_blob_def((5, 4), is_dynamic=True), x2 = flow.input_blob_def((5, 1), is_dynamic=True)):
    outputs = flow.dim0_dynamic_to_fixed([x1, x2])
    flow.watch(x1, Print("x1: "))
    flow.watch(x2, Print("x2: "))
    for i in range(len(outputs)):
        flow.watch(outputs[i], Print("out_" + str(i) + ": "))
    return outputs[0]

(TestJob(np.ones((3,4), dtype=np.float32), np.ones((3,1), dtype=np.float32)).get())

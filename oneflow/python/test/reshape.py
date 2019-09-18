import oneflow as flow
import numpy as np

flow.config.gpu_device_num(2)
flow.config.ctrl_port(12322)


@flow.function
def ReshapeJob0(x=flow.input_blob_def((10, 20, 20))):
    return flow.reshape(x, (200, 20))

random_array = np.random.rand(10, 20, 20).astype(np.float32)
np.array_equal(ReshapeJob0(random_array).get(),
               np.reshape(random_array, (200, 20)))

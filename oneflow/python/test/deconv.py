import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)


@flow.function
def DeconvJob(x=flow.input_blob_def((6, 3, 50, 50))):
    filter = flow.get_variable(name="filter", shape=(
        6, 3, 100, 100), dtype=flow.float32, initializer=flow.random_uniform_initializer())
    return flow.nn.conv2d_transpose(x, filter)


print(DeconvJob(np.random.randn(6, 3, 50, 50)).get())

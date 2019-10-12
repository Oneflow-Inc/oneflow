import oneflow as flow
import numpy as np

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float32)


@flow.function
def DeconvJob(x=flow.input_blob_def((64, 256, 14, 14))):
    filter = flow.get_variable(name="filter", shape=(
        256, 3, 2, 2), dtype=flow.float32, initializer=flow.random_uniform_initializer())
    return flow.nn.conv2d_transpose(x, filter, strides=2, data_format="NCHW")


print(DeconvJob(np.random.randn(64, 256, 14, 14).astype(np.float32)).get())

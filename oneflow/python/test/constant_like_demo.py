import oneflow as flow
import numpy as np
import oneflow.core.common.data_type_pb2 as data_type_util

flow.config.gpu_device_num(1)
flow.config.grpc_use_no_signal()

flow.config.default_data_type(flow.float)


@flow.function
def ConstantLikeJob(
    inputs=flow.input_blob_def(
        (2000, 2000), dtype=data_type_util.kFloat, is_dynamic=True
    ),
):
    return flow.constant_like(inputs, float(2.0))

output = ConstantLikeJob(np.ones((1024, 1024)).astype(np.float32)).get()
print(output.shape)
print(output)

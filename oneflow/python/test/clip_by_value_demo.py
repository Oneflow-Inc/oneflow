import tensorflow as tf
import oneflow as flow
import numpy as np
import oneflow.core.common.data_type_pb2 as data_type_util

import tensorflow as tf

tf.enable_eager_execution()
assert tf.executing_eagerly()

flow.config.gpu_device_num(1)
flow.config.grpc_use_no_signal()
flow.config.default_data_type(flow.float)


@flow.function
def ClipByValueJob(
    input_blob=flow.input_blob_def(
        (2000, 2000), dtype=data_type_util.kFloat, is_dynamic=True
    )
):
    return flow.clip_by_value(input_blob, clip_value_min=0.2, clip_value_max=0.8)


input_blob = tf.Variable(np.random.random_sample((1024, 1024)).astype(np.float32))
tf_out = tf.clip_by_value(input_blob, 0.2, 0.8)
of_out = ClipByValueJob(input_blob.numpy()).get()
print(np.max(np.absolute(tf_out - of_out)))

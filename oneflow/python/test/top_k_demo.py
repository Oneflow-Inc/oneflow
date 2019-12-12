import tensorflow as tf
import oneflow as flow
import numpy as np
import oneflow.core.common.data_type_pb2 as data_type_util

import tensorflow as tf
tf.enable_eager_execution()
assert(tf.executing_eagerly())

flow.config.gpu_device_num(1)
flow.config.grpc_use_no_signal()
flow.config.default_data_type(flow.float)

def test_top_k(static_shape, dynamic_shape, k):
    @flow.function
    def TopKJob(
        input_blob=flow.input_blob_def(
            static_shape, dtype=data_type_util.kFloat, is_dynamic=True
        )
    ):
      return flow.math.top_k(input_blob, k)
        
    input_blob = np.random.randint(1024, size=dynamic_shape).astype(np.float32)
    tf_out = tf.nn.top_k(tf.Variable(input_blob), k, sorted=True).indices.numpy()
    of_out = TopKJob(input_blob).get()
    print(np.max(np.absolute(tf_out - of_out)))
    np.allclose(tf_out, of_out)


test_top_k((1000,), (100,), 1)
# test_top_k((1, 1000,), (1, 100), 1)
# test_top_k((1024, 1024), (1000, 1000), 256)
# test_top_k((2048, 2048), (2000, 2000), 50)
# test_top_k((10, 10, 1000), (10, 10, 1000), 256)
# test_top_k((10, 10, 1000), (10, 10, 1000), 50)

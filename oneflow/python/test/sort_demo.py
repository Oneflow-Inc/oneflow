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

def test_sort(static_shape, dynamic_shape, direction):
    @flow.function
    def SortJob(
        input_blob=flow.input_blob_def(
            static_shape, dtype=data_type_util.kFloat, is_dynamic=True
        )
    ):
      return flow.sort(input_blob, direction)
        
    input_blob = np.random.randint(1024, size=dynamic_shape).astype(np.float32)
    tf_out = tf.sort(tf.Variable(input_blob), axis=-1, direction=direction).numpy()
    of_out = SortJob(input_blob).get()
    print(np.max(np.absolute(tf_out - of_out)))


test_sort((1000,), (100,), "ASCENDING")
# test_sort((1000,), (100,), "DESCENDING")
# test_sort((1, 1000,), (1, 100), "ASCENDING")
# test_sort((1, 1000,), (1, 100), "DESCENDING")
# test_sort((1024, 1024), (1000, 1000), "ASCENDING")
# test_sort((2048, 2048), (2000, 2000), "DESCENDING")
# test_sort((10, 10, 1000), (10, 10, 1000), "ASCENDING")
# test_sort((10, 10, 1000), (10, 10, 1000), "DESCENDING")

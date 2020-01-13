import tensorflow as tf
import oneflow as flow
import numpy as np
import oneflow.core.common.data_type_pb2 as data_type_util

flow.config.gpu_device_num(1)
flow.config.default_data_type(flow.float)

def test_top_k(static_shape, dynamic_shape, k):
    flow.clear_default_session()
    @flow.function
    def TopKJob(
        input_blob=flow.input_blob_def(
            static_shape, dtype=data_type_util.kFloat, is_dynamic=True
        )
    ):
      return flow.math.top_k(input_blob, k)

    input_blob = np.random.randint(1024, size=dynamic_shape).astype(np.float32)
    tf_out = tf.nn.top_k(tf.Variable(input_blob), k, sorted=True).indices.numpy()
    of_out = TopKJob(input_blob).get().ndarray()
    print(np.max(np.absolute(tf_out - of_out)))
    np.allclose(tf_out, of_out)


if __name__ == "__main__":
  test_top_k((2000,), (1024,), 1)
  test_top_k((2000,), (1024,), 64)
  test_top_k((2000,), (1024,), 128)
  test_top_k((2000,), (1024,), 256)
  test_top_k((1, 2000,), (1, 1024), 1)
  test_top_k((1, 2000,), (1, 1024), 64)
  test_top_k((1, 2000,), (1, 1024), 128)
  test_top_k((1, 2000,), (1, 1024), 256)
  test_top_k((2000, 2000), (1024, 1024), 32)
  test_top_k((2000, 2000), (1024, 1024), 64)
  test_top_k((2000, 2000), (1024, 1024), 128)
  test_top_k((2000, 2000), (1024, 1024), 256)
  test_top_k((10, 10, 2000), (10, 10, 1024), 32)
  test_top_k((10, 10, 2000), (10, 10, 1024), 64)
  test_top_k((10, 10, 2000), (10, 10, 1024), 128)
  test_top_k((10, 10, 2000), (10, 10, 1024), 256)

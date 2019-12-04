import tensorflow as tf
import oneflow as flow
import numpy as np

def compare_with_tensorflow(a_shape, b_shape, transpose_a=False, transpose_b=False):
    flow.clear_default_session()

    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

    @flow.function
    def MatmulTestJob(a=flow.input_blob_def(a_shape), b=flow.input_blob_def(b_shape)):
        return flow.matmul(a, b, transpose_a, transpose_b)

    a = np.random.random_sample(a_shape).astype(np.float32)
    b = np.random.random_sample(b_shape).astype(np.float32)
    # OneFlow
    of_out = MatmulTestJob(a, b).get()
    # TensorFlow
    tf_out = tf.matmul(tf.Variable(a), tf.Variable(b), transpose_a, transpose_b).numpy()
    assert np.allclose(of_out, tf_out)


def test_matmul(test_case):
    compare_with_tensorflow(a_shape=(512, 256), b_shape=(256, 1024))
    compare_with_tensorflow(a_shape=(256, 512), b_shape=(256, 1024), transpose_a=True)
    compare_with_tensorflow(a_shape=(512, 256), b_shape=(1024, 256), transpose_b=True)
    compare_with_tensorflow(
        a_shape=(256, 512), b_shape=(1024, 256), transpose_a=True, transpose_b=True
    )
    compare_with_tensorflow(a_shape=(10, 10, 64, 32), b_shape=(10, 10, 32, 128))
    compare_with_tensorflow(a_shape=(10, 10, 32, 64), b_shape=(10, 10, 32, 128), transpose_a=True)
    compare_with_tensorflow(a_shape=(10, 10, 64, 32), b_shape=(10, 10, 128, 32), transpose_b=True)
    compare_with_tensorflow(
        a_shape=(10, 10, 32, 64), b_shape=(10, 10, 128, 32), transpose_a=True, transpose_b=True
    )

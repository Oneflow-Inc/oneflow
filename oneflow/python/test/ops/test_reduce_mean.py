import tensorflow as tf
import oneflow as flow
import numpy as np


def compare_with_tensorflow(input_shape, axis=None, keepdims=False):
    flow.clear_default_session()

    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

    @flow.function
    def ReduceMeanTestJob(input=flow.input_blob_def(input_shape)):
        return flow.math.reduce_mean(input_tensor=input, axis=axis, keepdims=keepdims)

    input = np.random.random_sample(input_shape).astype(np.float32)
    # OneFlow
    of_out = ReduceMeanTestJob(input).get()
    # TensorFlow
    tf_out = tf.math.reduce_mean(tf.Variable(input), axis=axis, keepdims=keepdims).numpy()

    assert np.allclose(of_out, tf_out, atol=1e-7)

def test_reduce_mean(test_cast):
    compare_with_tensorflow(input_shape=(128, 128, 128), axis=(0, 2))
    compare_with_tensorflow(input_shape=(1024, 1024), axis=[1], keepdims=True)
    compare_with_tensorflow(input_shape=(128, 128, 128), axis=(0, 2), keepdims=False)

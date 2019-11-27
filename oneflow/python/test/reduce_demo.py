import tensorflow as tf
import oneflow as flow
import numpy as np

assert tf.executing_eagerly()


def test_reduce_sum(input_shape, axis=None, keepdims=False):
    input = np.random.random_sample(input_shape).astype(np.float32)

    # OneFlow
    flow.config.gpu_device_num(1)

    @flow.function
    def ReduceSumTestJob(input=flow.input_blob_def(input_shape)):
        return flow.math.reduce_sum(input_tensor=input, axis=axis, keepdims=keepdims)

    of_out = ReduceSumTestJob(input).get()

    # TensorFlow
    tf_out = tf.math.reduce_sum(
        tf.Variable(input), axis=axis, keepdims=keepdims
    ).numpy()

    assert np.allclose(of_out, tf_out, atol=1e-7)


def test_reduce_mean(input_shape, axis=None, keepdims=False):
    input = np.random.random_sample(input_shape).astype(np.float32)

    # OneFlow
    flow.config.gpu_device_num(1)

    @flow.function
    def ReduceMeanTestJob(input=flow.input_blob_def(input_shape)):
        return flow.math.reduce_mean(input_tensor=input, axis=axis, keepdims=keepdims)

    of_out = ReduceMeanTestJob(input).get()

    # TensorFlow
    tf_out = tf.math.reduce_mean(
        tf.Variable(input), axis=axis, keepdims=keepdims
    ).numpy()

    assert np.allclose(of_out, tf_out, atol=1e-7)


# run one example each time
if __name__ == "__main__":
     test_reduce_sum(input_shape=(128, 128, 128), axis=(0, 2))
    # test_reduce_sum(input_shape=(1024, 1024), axis=[1], keepdims=True)
    # test_reduce_sum(input_shape=(128, 128, 128), axis=(0, 2), keepdims=False)
    # test_reduce_mean(input_shape=(128, 128, 128), axis=(0, 2))

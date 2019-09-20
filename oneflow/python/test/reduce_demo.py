import tensorflow as tf
import oneflow as flow
import numpy as np

tf.enable_eager_execution()
assert tf.executing_eagerly()


def test_reduce_sum(input_shape, axis=None, keepdims=False):
    input = np.random.random_sample(input_shape).astype(np.float32)

    # OneFlow
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(1)
    flow.init(config)

    def ReduceSumTestJob(input=flow.input_blob_def(input_shape)):
        return flow.math.reduce_sum(input_tensor=input, axis=axis, keepdims=keepdims)

    flow.add_job(ReduceSumTestJob)
    with flow.Session() as sess:
        of_out = sess.run(ReduceSumTestJob, input).get()

    # TensorFlow
    tf_out = tf.math.reduce_sum(
        tf.Variable(input), axis=axis, keepdims=keepdims
    ).numpy()

    print("dense max diff: " + str(np.max(np.abs(of_out - tf_out))))
    assert np.allclose(of_out, tf_out, atol=1e-7)


def test_reduce_mean(input_shape, axis=None, keepdims=False):
    input = np.random.random_sample(input_shape).astype(np.float32)

    # OneFlow
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(1)
    flow.init(config)

    def ReduceMeanTestJob(input=flow.input_blob_def(input_shape)):
        return flow.math.reduce_mean(input_tensor=input, axis=axis, keepdims=keepdims)

    flow.add_job(ReduceMeanTestJob)
    with flow.Session() as sess:
        of_out = sess.run(ReduceMeanTestJob, input).get()

    # TensorFlow
    tf_out = tf.math.reduce_mean(
        tf.Variable(input), axis=axis, keepdims=keepdims
    ).numpy()

    print("dense max diff: " + str(np.max(np.abs(of_out - tf_out))))
    assert np.allclose(of_out, tf_out, atol=1e-7)


# run one example each time
if __name__ == "__main__":
    test_reduce_sum(input_shape=(1024, 1024))
    # test_reduce_sum(input_shape=(1024, 1024), axis=[1], keepdims=True)
    # test_reduce_sum(input_shape=(100, 100, 100), axis=[1, 2], keepdims=False)
    # test_reduce_mean(input_shape=(1024, 1024))

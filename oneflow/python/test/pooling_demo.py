import tensorflow as tf
import oneflow as flow
import numpy as np

tf.enable_eager_execution()
assert tf.executing_eagerly()


def test_max_pool_2d():
    x = np.random.random_sample((10, 1000, 1000, 3)).astype(np.float32)

    # OneFlow
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(1)
    flow.init(config)

    def MaxPool2DTestJob(x=flow.input_blob_def((10, 1000, 1000, 3))):
        job_conf = flow.get_cur_job_conf_builder()
        job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
        return flow.keras.pooling.max_pool_2d(
            x,
            pool_size=[10, 10],
            strides=[10, 10],
            padding="valid",
            data_format="channels_last",
        )

    flow.add_job(MaxPool2DTestJob)
    with flow.Session() as sess:
        of_out = sess.run(MaxPool2DTestJob, x).get()

    # TensorFlow
    tf_max_pooling_2d = tf.keras.layers.MaxPooling2D(
        pool_size=[10, 10],
        strides=[10, 10],
        padding="valid",
        data_format="channels_last",
    )
    tf_out = tf_max_pooling_2d(tf.Variable(x)).numpy()

    max_diff = np.max(np.abs(of_out - tf_out))
    print("MaxPool2D max diff: " + str(max_diff))


def test_average_pool_2d():
    x = np.random.random_sample((10, 1000, 1000, 3)).astype(np.float32)

    # OneFlow
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(1)
    flow.init(config)

    def AveragePool2DTestJob(x=flow.input_blob_def((10, 1000, 1000, 3))):
        job_conf = flow.get_cur_job_conf_builder()
        job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
        return flow.keras.pooling.average_pool_2d(
            x,
            pool_size=[10, 10],
            strides=[10, 10],
            padding="valid",
            data_format="channels_last",
        )

    flow.add_job(AveragePool2DTestJob)
    with flow.Session() as sess:
        of_out = sess.run(AveragePool2DTestJob, x).get()

    # TensorFlow
    tf_average_pooling_2d = tf.keras.layers.AveragePooling2D(
        pool_size=[10, 10],
        strides=[10, 10],
        padding="valid",
        data_format="channels_last",
    )
    tf_out = tf_average_pooling_2d(tf.Variable(x)).numpy()

    max_diff = np.max(np.abs(of_out - tf_out))
    print("AveragePool2D max diff: " + str(max_diff))


if __name__ == "__main__":
    test_max_pool_2d()
    test_average_pool_2d()

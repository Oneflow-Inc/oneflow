import tensorflow as tf
import oneflow as flow
import numpy as np

tf.enable_eager_execution()
assert tf.executing_eagerly()


def test_max_pool2d():
    input = np.random.random_sample((10, 1000, 1000, 3)).astype(np.float32)

    # OneFlow
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(1)
    flow.init(config)

    def MaxPool2DTestJob(input=flow.input_blob_def((10, 1000, 1000, 3))):
        job_conf = flow.get_cur_job_conf_builder()
        job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
        return flow.nn.max_pool2d(
            input, ksize=[10, 10], strides=[10], padding="VALID", data_format="NHWC"
        )

    flow.add_job(MaxPool2DTestJob)
    with flow.Session() as sess:
        of_out = sess.run(MaxPool2DTestJob, input).get()

    # TensorFlow
    tf_max_pooling_2d = tf.layers.MaxPooling2D(
        pool_size=[10, 10],
        strides=[10, 10],
        padding="valid",
        data_format="channels_last",
    )
    tf_out = tf_max_pooling_2d(tf.Variable(input)).numpy()

    max_diff = np.max(np.abs(of_out - tf_out))
    print("MaxPool2D max diff: " + str(max_diff))


def test_avg_pool2d():
    input = np.random.random_sample((10, 1000, 1000, 3)).astype(np.float32)

    # OneFlow
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(1)
    flow.init(config)

    def AveragePool2DTestJob(input=flow.input_blob_def((10, 1000, 1000, 3))):
        job_conf = flow.get_cur_job_conf_builder()
        job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
        return flow.nn.avg_pool2d(
            input, ksize=[10], strides=[10, 10], padding="VALID", data_format="NHWC"
        )

    flow.add_job(AveragePool2DTestJob)
    with flow.Session() as sess:
        of_out = sess.run(AveragePool2DTestJob, input).get()

    # TensorFlow
    tf_average_pooling_2d = tf.layers.AveragePooling2D(
        pool_size=[10, 10],
        strides=[10, 10],
        padding="valid",
        data_format="channels_last",
    )
    tf_out = tf_average_pooling_2d(tf.Variable(input)).numpy()

    max_diff = np.max(np.abs(of_out - tf_out))
    print("AveragePool2D max diff: " + str(max_diff))


def test_max_pool3d():
    input = np.random.random_sample((10, 100, 100, 100, 3)).astype(np.float32)

    # OneFlow
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(1)
    flow.init(config)

    def MaxPool3DTestJob(input=flow.input_blob_def((10, 100, 100, 100, 3))):
        job_conf = flow.get_cur_job_conf_builder()
        job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
        return flow.nn.max_pool3d(
            input,
            ksize=[10, 10, 10],
            strides=[10],
            padding="VALID",
            data_format="NDHWC",
        )

    flow.add_job(MaxPool3DTestJob)
    with flow.Session() as sess:
        of_out = sess.run(MaxPool3DTestJob, input).get()

    # TensorFlow
    tf_max_pooling_3d = tf.layers.MaxPooling3D(
        pool_size=[10, 10, 10],
        strides=[10, 10, 10],
        padding="valid",
        data_format="channels_last",
    )
    tf_out = tf_max_pooling_3d(tf.Variable(input)).numpy()

    max_diff = np.max(np.abs(of_out - tf_out))
    print("MaxPool3D max diff: " + str(max_diff))


def test_avg_pool3d():
    input = np.random.random_sample((10, 100, 100, 100, 3)).astype(np.float32)

    # OneFlow
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(1)
    flow.init(config)

    def AveragePool3DTestJob(input=flow.input_blob_def((10, 100, 100, 100, 3))):
        job_conf = flow.get_cur_job_conf_builder()
        job_conf.batch_size(10).data_part_num(1).default_data_type(flow.float)
        return flow.nn.avg_pool3d(
            input,
            ksize=[10],
            strides=10,
            padding="VALID",
            data_format="NDHWC",
        )

    flow.add_job(AveragePool3DTestJob)
    with flow.Session() as sess:
        of_out = sess.run(AveragePool3DTestJob, input).get()

    # TensorFlow
    tf_average_pooling_3d = tf.layers.AveragePooling3D(
        pool_size=[10, 10, 10],
        strides=[10, 10, 10],
        padding="valid",
        data_format="channels_last",
    )
    tf_out = tf_average_pooling_3d(tf.Variable(input)).numpy()

    max_diff = np.max(np.abs(of_out - tf_out))
    print("AveragePool3D max diff: " + str(max_diff))


# run one example each time
if __name__ == "__main__":
    test_max_pool2d()
    # test_avg_pool2d()
    # test_max_pool3d()
    # test_avg_pool3d()

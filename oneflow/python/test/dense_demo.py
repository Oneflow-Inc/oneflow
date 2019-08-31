import tensorflow as tf
import oneflow as flow
import numpy as np

tf.enable_eager_execution()
assert tf.executing_eagerly()

of_activation_map = {"sigmoid": flow.keras.activations.sigmoid}
tf_activation_map = {"sigmoid": tf.nn.sigmoid}


def test_dense(
    in_shape, units, activation=None, use_bias=True, trainable=True, name=None
):
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(1)
    flow.init(config)

    input = np.random.random_sample((64, 32)).astype(np.float32)

    # OneFlow
    def DenseTestJob(inputs=flow.input_blob_def((64, 32))):
        job_conf = flow.get_cur_job_conf_builder()
        job_conf.batch_size(1).data_part_num(1).default_data_type(flow.float)
        return flow.layers.dense(
            inputs=inputs, units=units, activation=of_activation_map["sigmoid"]
        )

    flow.add_job(DenseTestJob)

    ckp = flow.train.CheckPoint()
    status = ckp.restore()
    with flow.Session() as sess:
        status.initialize_or_restore(session=sess)
        of_out = sess.run(DenseTestJob, input).get()

    # TensorFlow
    tf_out = tf.layers.dense(
        tf.Variable(input),
        units,
        tf_activation_map["sigmoid"],
        use_bias=use_bias,
        kernel_initializer=tf.constant_initializer(),
    ).numpy()

    print("matmul max diff: " + str(np.max(np.abs(of_out - tf_out))))


# run one example each time
if __name__ == "__main__":
    test_dense(
        in_shape=(64, 32),
        units=128,
        activation="sigmoid",
        use_bias=False,
        trainable=False,
    )

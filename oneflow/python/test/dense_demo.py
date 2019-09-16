import tensorflow as tf
import oneflow as flow
import numpy as np

tf.enable_eager_execution()
assert tf.executing_eagerly()

of_activation_map = {"sigmoid": flow.keras.activations.sigmoid, "none": None}
tf_activation_map = {"sigmoid": tf.nn.sigmoid, "none": None}


def test_dense(in_shape, units, activation=None, use_bias=True):
    config = flow.ConfigProtoBuilder()
    config.gpu_device_num(1)
    flow.init(config)

    input = np.random.random_sample(in_shape).astype(np.float32) / 10000

    # OneFlow
    @flow.function
    def DenseTestJob(inputs=flow.input_blob_def(in_shape)):
        flow.config.default_data_type(flow.float)
        return flow.layers.dense(
            inputs=inputs,
            units=units,
            activation=of_activation_map[activation],
            use_bias=use_bias,
            kernel_initializer=flow.constant_initializer(value=1),
            bias_initializer=flow.constant_initializer(value=1),
        )

    check_point = flow.train.SimpleCheckPointManager("model_save")
    check_point.initialize_or_restore()
    of_out = DenseTestJob(input).get()

    # TensorFlow
    tf_out = tf.layers.dense(
        tf.Variable(input),
        units,
        tf_activation_map[activation],
        use_bias=use_bias,
        kernel_initializer=tf.ones_initializer(),
        bias_initializer=tf.ones_initializer(),
    ).numpy()

    print("dense max diff: " + str(np.max(np.abs(of_out - tf_out))))
    assert np.allclose(of_out, tf_out, atol=1e-7)


# run one example each time
if __name__ == "__main__":
    test_dense(in_shape=(1024, 2048), units=512, activation="sigmoid", use_bias=False)
    # test_dense(in_shape=(16, 32, 64, 128), units=512, activation="sigmoid", use_bias=False)
    # test_dense(in_shape=(1024, 2048), units=512, activation="none", use_bias=True)
    # test_dense(in_shape=(16, 32, 64, 128), units=512, activation="sigmoid", use_bias=True)

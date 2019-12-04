import tensorflow as tf
import oneflow as flow
import numpy as np

of_activation_map = {"sigmoid": flow.keras.activations.sigmoid, "none": None}
tf_activation_map = {"sigmoid": tf.nn.sigmoid, "none": None}


def conpare_with_tensorflow(in_shape, units, activation=None, use_bias=True):
    flow.clear_default_session()

    flow.config.gpu_device_num(1)
    flow.config.default_data_type(flow.float)

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

    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = DenseTestJob(input).get()

    # TensorFlow
    tf_out = tf.compat.v1.layers.dense(
        tf.Variable(input),
        units,
        tf_activation_map[activation],
        use_bias=use_bias,
        kernel_initializer=tf.ones_initializer(),
        bias_initializer=tf.ones_initializer(),
    ).numpy()

    assert np.allclose(of_out, tf_out, atol=1e-7)


def test_dense(test_case):
    conpare_with_tensorflow(in_shape=(1024, 2048), units=512, activation="sigmoid", use_bias=False)
    conpare_with_tensorflow(
        in_shape=(16, 32, 64, 128), units=512, activation="sigmoid", use_bias=False
    )
    conpare_with_tensorflow(in_shape=(1024, 2048), units=512, activation="none", use_bias=True)
    conpare_with_tensorflow(
        in_shape=(16, 32, 64, 128), units=512, activation="sigmoid", use_bias=True
    )

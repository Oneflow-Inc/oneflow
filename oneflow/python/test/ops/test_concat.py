import numpy as np
import tensorflow as tf
import oneflow as flow


def test_concat(test_case):
    flow.config.gpu_device_num(2)
    flow.config.default_data_type(flow.float)

    @flow.function
    def ConcatJob(a=flow.input_blob_def((5, 2)), b=flow.input_blob_def((5, 3))):
        return flow.concat([a, b], axis=1)

    a = np.arange(10, dtype=np.float32).reshape(5, 2)
    b = np.arange(15, dtype=np.float32).reshape(5, 3)
    # OneFlow
    of_out = ConcatJob(a, b).get()
    # TensorFlow
    tf_out = tf.concat([a, b], 1)
    test_case.assertTrue(np.allclose(of_out, tf_out))

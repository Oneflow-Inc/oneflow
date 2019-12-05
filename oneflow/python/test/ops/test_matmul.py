import tensorflow as tf
import oneflow as flow
import numpy as np
from test_util import GenCartesianProduct


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


def filter_args(arg_list):
    def trans_shape(shape):
        tmp_shape = shape[:-2]
        tmp_shape += (shape[-1], shape[-2])
        return tmp_shape

    ret = []
    for arg in arg_list:
        a_shape = arg[0]
        b_shape = arg[1]
        if arg[2]:  # transpose_a
            a_shape = trans_shape(a_shape)
        if arg[3]:  # transpose_b
            b_shape = trans_shape(b_shape)
        if a_shape[-1] == b_shape[-2]:
            ret.append(tuple(arg))
    return ret

def gen_arg_list():
    matmul_args = [
        [(512, 256), (256, 512)],
        [(256, 1024), (1024, 256)],
        [True, False],
        [True, False],
    ]
    matmul_args = filter_args(GenCartesianProduct(matmul_args))

    batch_matmul_args = [
        [(10, 10, 64, 32), (10, 10, 32, 64)],
        [(10, 10, 32, 128), (10, 10, 128, 32)],
        [True, False],
        [True, False],
    ]
    batch_matmul_args = filter_args(GenCartesianProduct(batch_matmul_args))

    return matmul_args + batch_matmul_args


def test_matmul(test_case):
    for arg in gen_arg_list():
        compare_with_tensorflow(*arg)

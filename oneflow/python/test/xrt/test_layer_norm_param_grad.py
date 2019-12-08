import unittest
import numpy as np

import oneflow as flow

def make_job(shape, gamma_shape, params_axis, dtype=flow.float32):
    @flow.function
    def layer_norm_param_grad_job(dy = flow.input_blob_def(shape, dtype=dtype),
                                  norm = flow.input_blob_def(shape, dtype=dtype),
                                  gamma = flow.input_blob_def(gamma_shape, dtype=dtype)):
        flow.config.use_xla_jit(False)
        flow.config.use_tensorrt(False)
        return flow.layers.layer_norm_param_grad(
            dy, norm, gamma, begin_params_axis=params_axis)
    return layer_norm_param_grad_job

def make_xla_job(shape, gamma_shape, params_axis, dtype=flow.float32):
    @flow.function
    def xla_layer_norm_param_grad_job(dy = flow.input_blob_def(shape, dtype=dtype),
                                      norm = flow.input_blob_def(shape, dtype=dtype),
                                      gamma = flow.input_blob_def(gamma_shape, dtype=dtype)):
        flow.config.use_xla_jit(True)
        flow.config.use_tensorrt(False)
        return flow.layers.layer_norm_param_grad(
            dy, norm, gamma, begin_params_axis=params_axis)
    return xla_layer_norm_param_grad_job

class TestLayerNormParamGrad(unittest.TestCase):
    def _test_body(self, dy, norm, gamma, params_axis,
                   dtype=np.float32):
        f1 = make_job(dy.shape, gamma.shape, params_axis, dtype=flow.float32)
        f2 = make_xla_job(dy.shape, gamma.shape, params_axis, dtype=flow.float32)
        (d_norm1, d_beta1, d_gamma1) = f1(dy, norm, gamma).get()
        (d_norm2, d_beta2, d_gamma2) = f2(dy, norm, gamma).get()

        print("normalize diff:")
        print("    without xla: ", d_norm1)
        print("    with xla: ", d_norm2)
        print("beta diff:")
        print("    without xla: ", d_beta1)
        print("    with xla: ", d_beta2)
        print("gamma diff:")
        print("    without xla: ", d_gamma1)
        print("    with xla: ", d_gamma2)

        self.assertTrue(d_norm1.shape, d_norm2.shape)
        self.assertTrue(d_beta1.shape, d_beta2.shape)
        self.assertTrue(d_gamma1.shape, d_gamma2.shape)

        self.assertTrue(np.allclose(d_norm1, d_norm2, rtol=1e-03, atol=1e-05))
        self.assertTrue(np.allclose(d_beta1, d_beta2, rtol=1e-03, atol=1e-05))
        self.assertTrue(np.allclose(d_gamma1, d_gamma2, rtol=1e-03, atol=1e-05))

        flow.clear_default_session()

    def _test_ones_body(self, shape,
                        params_axis=-1,
                        dtype=np.float32):
        dy = np.ones(shape, dtype=dtype)
        norm = np.ones(shape, dtype=dtype)
        if params_axis < 0:
            params_axis += len(shape)
        gamma_shape = shape[params_axis:]
        if len(gamma_shape) == 0:
          gamma_shape = [1]
        gamma = np.ones(gamma_shape, dtype=dtype)
        self._test_body(dy, norm, gamma, params_axis, dtype=dtype)

    def _test_random_body(self, shape,
                          params_axis=-1,
                          dtype=np.float32):
        dy = np.random.random(shape).astype(dtype)
        norm = np.random.random(shape).astype(dtype)
        if params_axis < 0:
            params_axis += len(shape)
        gamma_shape = shape[params_axis:]
        if len(gamma_shape) == 0:
          gamma_shape = [1]
        gamma = np.random.random(gamma_shape).astype(dtype)
        self._test_body(dy, norm, gamma, params_axis, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1, 10))
        self._test_ones_body((2, 10, 2))
        self._test_ones_body((2, 5, 2, 2))

    def test_random_input(self):
        self._test_random_body((1, 10))
        self._test_random_body((2, 10, 2))
        self._test_random_body((2, 5, 2, 2))

if __name__ == '__main__':
    unittest.main()

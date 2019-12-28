import unittest
import numpy as np

import oneflow as flow

config = flow.function_config()

def make_job(input_shape, norm_axis, params_axis, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.function(config)
    def layer_norm_job(x = flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.layers.layer_norm(x, begin_norm_axis=norm_axis,
                                      begin_params_axis=params_axis)
    return layer_norm_job

def make_xla_job(input_shape, norm_axis, params_axis, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.function(config)
    def xla_layer_norm_job(x = flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.layers.layer_norm(x, begin_norm_axis=norm_axis,
                                      begin_params_axis=params_axis)
    return xla_layer_norm_job

class TestLayerNorm(unittest.TestCase):
    def _test_body(self, x, norm_axis, params_axis, dtype=np.float32):
        f1 = make_job(x.shape, norm_axis, params_axis, dtype=flow.float32)
        f2 = make_xla_job(x.shape, norm_axis, params_axis, dtype=flow.float32)
        a = f1(x).get()
        b = f2(x).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape,
                        norm_axis=-1,
                        params_axis=-1,
                        dtype=np.float32):
        x = np.ones(shape, dtype=dtype)
        self._test_body(x, norm_axis, params_axis, dtype=dtype)

    def _test_random_body(self, shape,
                          norm_axis=-1,
                          params_axis=-1,
                          dtype=np.float32):
        x = (10 * np.random.random(shape)).astype(dtype)
        self._test_body(x, norm_axis, params_axis, dtype=dtype)

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

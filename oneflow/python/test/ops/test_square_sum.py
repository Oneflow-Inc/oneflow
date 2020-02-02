import oneflow as flow
import numpy as np

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def _check(test_case, x, y):
    ref_y = np.array(np.sum(x ** 2))
    test_case.assertTrue(np.allclose(y, ref_y))


def _run_test(test_case, x, dtype, device):
    @flow.function(func_config)
    def SquareSum(x=flow.FixedTensorDef(x.shape, dtype=dtype)):
        with flow.fixed_placement(device, "0:0"):
            return flow.experimental.square_sum(x)

    y = SquareSum(x).get()
    _check(test_case, x, y.ndarray())


def test_square_sum_random_gpu(test_case):
    x = np.random.uniform(-0.01, 0.01, (64, 64)).astype(np.float32)
    _run_test(test_case, x, flow.float32, 'gpu')


def test_square_sum_small_blob_gpu(test_case):
    x = np.random.uniform(-0.01, 0.01, (64,)).astype(np.float32)
    _run_test(test_case, x, flow.float32, 'gpu')

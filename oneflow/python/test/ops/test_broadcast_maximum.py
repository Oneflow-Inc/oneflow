import numpy as np
import oneflow as flow

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def _check(test_case, a, b, out):
    test_case.assertTrue(np.array_equal(np.maximum(a, b), out))


def _run_test(test_case, a, b, dtype, device):
    @flow.global_function(func_config)
    def BroadcastMaximum(
        a=flow.FixedTensorDef(a.shape, dtype=dtype),
        b=flow.FixedTensorDef(b.shape, dtype=dtype),
    ):
        with flow.fixed_placement(device, "0:0"):
            return flow.math.maximum(a, b)

    out = BroadcastMaximum(a, b).get()
    _check(test_case, a, b, out.ndarray())


def test_broadcast_maximum_random_gpu(test_case):
    a = np.random.rand(1024, 1024).astype(np.float32)
    b = np.random.rand(1024, 1024).astype(np.float32)
    _run_test(test_case, a, b, flow.float32, "gpu")


def test_broadcast_maximum_broadcast_gpu(test_case):
    a = np.random.rand(1024, 1).astype(np.float32)
    b = np.random.rand(1, 1024).astype(np.float32)
    _run_test(test_case, a, b, flow.float32, "gpu")

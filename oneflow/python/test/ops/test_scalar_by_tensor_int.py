import numpy as np
import oneflow as flow


def _check(test_case, x, y, out, case):
    if case == "add":
        np_out = np.add(x, y)
    elif case == "sub":
        np_out = np.subtract(x, y)
    elif case == "mul":
        np_out = np.multiply(x, y)
    elif case == "div":
        if type(y[0]) == np.float32 or type(y[0]) == np.double:
            np_out = np.divide(x, y)
        else:
            np_out = np.floor_divide(x, y)

    test_case.assertTrue(np.allclose(np_out, out, rtol=1e-5, atol=1e-5))


def _run_test(test_case, x, y, case, dtype=None, device="gpu"):
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.consistent_strategy())

    @flow.global_function(func_config)
    def ScalarByTensorJob(
        x=flow.FixedTensorDef(x.shape, dtype=dtype),
        y=flow.FixedTensorDef(y.shape, dtype=dtype),
    ):
        if case == "add":
            return flow.math.add(x, y)
        elif case == "sub":
            return flow.math.subtract(x, y)
        elif case == "mul":
            return flow.math.multiply(x, y)
        elif case == "div":
            return flow.math.divide(x, y)

    out = ScalarByTensorJob(x, y).get()
    _check(test_case, x, y, out.ndarray(), case)


def test_scalar_add_by_tensor_gpu_float(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
    y = np.random.rand(1).astype(np.float32)
    _run_test(test_case, x, y, "add", flow.float, "gpu")


def test_scalar_add_by_tensor_cpu_float(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
    y = np.random.rand(1).astype(np.float32)
    _run_test(test_case, x, y, "add", flow.float, "cpu")


def test_scalar_add_by_tensor_gpu_double(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.double)
    y = np.random.rand(1).astype(np.double)
    _run_test(test_case, x, y, "add", flow.double, "gpu")


def test_scalar_add_by_tensor_cpu_double(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.double)
    y = np.random.rand(1).astype(np.double)
    _run_test(test_case, x, y, "add", flow.double, "cpu")


def test_scalar_add_by_tensor_gpu_int8(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
    _run_test(test_case, x, y, "add", flow.int8, "gpu")


def test_scalar_add_by_tensor_cpu_int8(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
    _run_test(test_case, x, y, "add", flow.int8, "cpu")


def test_scalar_add_by_tensor_gpu_int32(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
    _run_test(test_case, x, y, "add", flow.int32, "gpu")


def test_scalar_add_by_tensor_cpu_int32(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
    _run_test(test_case, x, y, "add", flow.int32, "cpu")


def test_scalar_add_by_tensor_gpu_int64(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
    _run_test(test_case, x, y, "add", flow.int64, "gpu")


def test_scalar_add_by_tensor_cpu_int64(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
    _run_test(test_case, x, y, "add", flow.int64, "cpu")


def test_scalar_sub_by_tensor_gpu_float(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
    y = np.random.rand(1).astype(np.float32)
    _run_test(test_case, x, y, "sub", flow.float, "gpu")


def test_scalar_sub_by_tensor_cpu_float(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
    y = np.random.rand(1).astype(np.float32)
    _run_test(test_case, x, y, "sub", flow.float, "cpu")


def test_scalar_sub_by_tensor_gpu_double(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.double)
    y = np.random.rand(1).astype(np.double)
    _run_test(test_case, x, y, "sub", flow.double, "gpu")


def test_scalar_sub_by_tensor_cpu_double(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.double)
    y = np.random.rand(1).astype(np.double)
    _run_test(test_case, x, y, "sub", flow.double, "cpu")


def test_scalar_sub_by_tensor_gpu_int8(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
    _run_test(test_case, x, y, "sub", flow.int8, "gpu")


def test_scalar_sub_by_tensor_cpu_int8(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
    _run_test(test_case, x, y, "sub", flow.int8, "cpu")


def test_scalar_sub_by_tensor_gpu_int32(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
    _run_test(test_case, x, y, "sub", flow.int32, "gpu")


def test_scalar_sub_by_tensor_cpu_int32(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
    _run_test(test_case, x, y, "sub", flow.int32, "cpu")


def test_scalar_sub_by_tensor_gpu_int64(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
    _run_test(test_case, x, y, "sub", flow.int64, "gpu")


def test_scalar_sub_by_tensor_cpu_int64(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
    _run_test(test_case, x, y, "sub", flow.int64, "cpu")


def test_scalar_mul_by_tensor_gpu_float(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
    y = np.random.rand(1).astype(np.float32)
    _run_test(test_case, x, y, "mul", flow.float, "gpu")


def test_scalar_mul_by_tensor_cpu_float(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
    y = np.random.rand(1).astype(np.float32)
    _run_test(test_case, x, y, "mul", flow.float, "cpu")


def test_scalar_mul_by_tensor_gpu_double(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.double)
    y = np.random.rand(1).astype(np.double)
    _run_test(test_case, x, y, "mul", flow.double, "gpu")


def test_scalar_mul_by_tensor_cpu_double(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.double)
    y = np.random.rand(1).astype(np.double)
    _run_test(test_case, x, y, "mul", flow.double, "cpu")


def test_scalar_mul_by_tensor_gpu_int8(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
    _run_test(test_case, x, y, "mul", flow.int8, "gpu")


def test_scalar_mul_by_tensor_cpu_int8(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
    _run_test(test_case, x, y, "mul", flow.int8, "cpu")


def test_scalar_mul_by_tensor_gpu_int32(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
    _run_test(test_case, x, y, "mul", flow.int32, "gpu")


def test_scalar_mul_by_tensor_cpu_int32(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
    _run_test(test_case, x, y, "mul", flow.int32, "cpu")


def test_scalar_mul_by_tensor_gpu_int64(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
    _run_test(test_case, x, y, "mul", flow.int64, "gpu")


def test_scalar_mul_by_tensor_cpu_int64(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
    _run_test(test_case, x, y, "mul", flow.int64, "cpu")


def test_scalar_div_by_tensor_gpu_float(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
    y = np.random.rand(1).astype(np.float32)
    _run_test(test_case, x, y, "div", flow.float, "gpu")


def test_scalar_div_by_tensor_cpu_float(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.float32)
    y = np.random.rand(1).astype(np.float32)
    _run_test(test_case, x, y, "div", flow.float, "cpu")


def test_scalar_div_by_tensor_gpu_double(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.double)
    y = np.random.rand(1).astype(np.double)
    _run_test(test_case, x, y, "div", flow.double, "gpu")


def test_scalar_div_by_tensor_cpu_double(test_case):
    x = np.random.rand(10, 3, 32, 1024).astype(np.double)
    y = np.random.rand(1).astype(np.double)
    _run_test(test_case, x, y, "div", flow.double, "cpu")


def test_scalar_div_by_tensor_gpu_int8(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
    _run_test(test_case, x, y, "div", flow.int8, "gpu")


def test_scalar_div_by_tensor_cpu_int8(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int8)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int8)
    _run_test(test_case, x, y, "div", flow.int8, "cpu")


def test_scalar_div_by_tensor_gpu_int32(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
    _run_test(test_case, x, y, "div", flow.int32, "gpu")


def test_scalar_div_by_tensor_cpu_int32(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int32)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int32)
    _run_test(test_case, x, y, "div", flow.int32, "cpu")


def test_scalar_div_by_tensor_gpu_int64(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
    _run_test(test_case, x, y, "div", flow.int64, "gpu")


def test_scalar_div_by_tensor_cpu_int64(test_case):
    x = np.random.randint(low=1, high=10, size=(10, 3, 32, 1024), dtype=np.int64)
    y = np.random.randint(low=1, high=10, size=(1,), dtype=np.int64)
    _run_test(test_case, x, y, "div", flow.int64, "cpu")

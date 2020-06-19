import numpy as np
import oneflow as flow


def test_get_variable_with_same_name(test_case):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    def get_v():
        return flow.get_variable(
            name="var",
            shape=(5, 2),
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(),
        )

    @flow.global_function(func_config)
    def TestJob0():
        v1 = get_v()
        v2 = get_v()
        return v1, v2

    @flow.global_function(func_config)
    def TestJob1():
        return get_v()

    check_point = flow.train.CheckPoint()
    check_point.init()
    j0_v1, j0_v2 = TestJob0().get()
    j1_v = TestJob1().get()
    test_case.assertTrue(np.array_equal(j0_v1.ndarray(), j0_v2.ndarray()))
    test_case.assertTrue(np.array_equal(j0_v1.ndarray(), j1_v.ndarray()))

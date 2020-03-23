import oneflow as flow
import numpy as np


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

    @flow.function(func_config)
    def TestJob0():
        v1 = get_v()
        return v1

    @flow.function(func_config)
    def TestJob1():
        v1 = get_v()
        return v1

    check_point = flow.train.CheckPoint()
    check_point.init()
    j0_v1 = TestJob0().get().ndarray()
    j1_v1 = TestJob1().get().ndarray()
    test_case.assertTrue(np.array_equal(j0_v1, j1_v1))

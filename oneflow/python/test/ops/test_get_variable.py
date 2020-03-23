import oneflow as flow
import numpy as np


def test_get_variable_with_same_name(test_case):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    def get_v(random_seed=None):
        return flow.get_variable(
            name="var",
            shape=(5, 2),
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(),
            random_seed=random_seed,
        )

    @flow.function(func_config)
    def TestJob0():
        return get_v()

    @flow.function(func_config)
    def TestJob1():
        return get_v()

    check_point = flow.train.CheckPoint()
    check_point.init()
    j0_var = TestJob0().get().ndarray()
    j1_var = TestJob1().get().ndarray()
    test_case.assertTrue(np.array_equal(j0_var, j1_var))

import oneflow as flow
import numpy as np


def test_name_scope(test_case):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    def get_var(var_name):
        return flow.get_variable(
            name=var_name,
            shape=(2, 256, 14, 14),
            dtype=flow.float32,
            initializer=flow.random_uniform_initializer(),
        )

    @flow.function(func_config)
    def test_name_scope_job():
        with flow.experimental.name_scope("backbone"):
            with flow.experimental.name_scope("branch"):
                var1 = get_var("var")

            with flow.experimental.name_scope("branch"):
                var2 = get_var("var")

        var3 = get_var("backbone-branch-var")
        return var1, var2, var3

    check_point = flow.train.CheckPoint()
    check_point.init()
    var1, var2, var3 = test_name_scope_job().get()
    test_case.assertTrue(np.array_equal(var1.ndarray(), var2.ndarray()))
    test_case.assertTrue(np.array_equal(var1.ndarray(), var3.ndarray()))

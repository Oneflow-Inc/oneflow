import oneflow as flow
import numpy as np
import os


# def test_lazy_output(test_case):
#     flow.clear_default_session()
#     flow.enable_eager_execution(False)
#     func_config = flow.FunctionConfig()
#     func_config.default_data_type(flow.float)
#     func_config.default_placement_scope(flow.device_prior_placement("gpu", "0:0"))

#     @flow.global_function(func_config)
#     def foo_job():
#         x = flow.constant(1, shape=(2, 5), dtype=flow.float)
#         return x

#     ret = foo_job().get()
#     print(type(ret))


def test_eager_output(test_case):
    if os.getenv("ENABLE_USER_OP") != "True":
        return

    # flow.clear_default_session()
    flow.enable_eager_execution()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    # func_config.default_placement_scope(flow.device_prior_placement("gpu", "0:0"))

    @flow.global_function(func_config)
    def foo_job():
        x = flow.constant(1, shape=(2, 5), dtype=flow.float)
        # print(x.numpy_mirrored_list())
        return x

    # foo_job()
    ret = foo_job().get()
    test_case.assertTrue(
        np.array_equal(np.ones(shape=(2, 5), dtype=np.single), ret.ndarray_list()[0])
    )

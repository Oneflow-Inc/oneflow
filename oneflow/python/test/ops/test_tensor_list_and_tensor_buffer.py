import numpy as np
import oneflow as flow


def _of_tensor_list_identity(test_case, verbose=False):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.global_function(func_config)
    def job_fn(x_def=flow.MirroredTensorListDef(shape=(2, 5))):
        x = flow.identity(x_def)
        return x

    input_1 = np.random.rand(1, 3).astype(np.float32)
    input_2 = np.random.rand(1, 4).astype(np.float32)

    ret = job_fn([[input_1, input_2]]).get()
    ret_arr_list = ret.ndarray_lists()

    if verbose:
        print("input_1 =", input_1)
        print("input_2 =", input_2)
        print("ret_arr_list =", ret_arr_list)

    test_case.assertTrue(np.array_equal(input_1, ret_arr_list[0][0]))
    test_case.assertTrue(np.array_equal(input_2, ret_arr_list[0][1]))


def _of_tensor_list_to_tensor_buffer(test_case, verbose=False):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_distribute_strategy(flow.distribute.mirrored_strategy())

    @flow.global_function(func_config)
    def job_fn(x_def=flow.MirroredTensorListDef(shape=(2, 5, 4), dtype=flow.float)):
        x = flow.tensor_list_to_tensor_buffer(x_def)
        return flow.tensor_buffer_to_tensor_list(x, shape=(5, 4), dtype=flow.float)

    input_1 = np.random.rand(1, 3, 4).astype(np.float32)
    input_2 = np.random.rand(1, 2, 4).astype(np.float32)
    ret = job_fn([[input_1, input_2]]).get()
    ret_arr_list = ret.ndarray_lists()

    if verbose:
        print("input_1 =", input_1)
        print("input_2 =", input_2)
        print("ret_arr_list =", ret_arr_list)

    test_case.assertTrue(np.array_equal(input_1, ret_arr_list[0][0]))
    test_case.assertTrue(np.array_equal(input_2, ret_arr_list[0][1]))


def test_tensor_list_input_output(test_case):
    _of_tensor_list_identity(test_case)


def test_tensor_list_and_tensor_buffer_conversion(test_case):
    _of_tensor_list_to_tensor_buffer(test_case)

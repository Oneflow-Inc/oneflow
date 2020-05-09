import oneflow as flow
import numpy as np


def _of_tensor_list_to_tensor_buffer():
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def job_fn(input_def=flow.MirroredTensorListDef(shape=(2, 5))):
        with flow.fixed_placement("cpu", "0:0"):
            x = flow.tensor_list_to_tensor_buffer(input_def)
            return flow.tensor_buffer_to_tensor_list(x, shape=(2, 5), dtype=flow.float)

    input_1 = np.random.rand(1, 3).astype(np.float32)
    input_2 = np.random.rand(1, 4).astype(np.float32)
    ret = job_fn([[input_1, input_2]]).get().ndarray_lists()
    print(ret)


def test_tensor_list_and_tensor_buffer_conversion(test_case):
    _of_tensor_list_to_tensor_buffer()

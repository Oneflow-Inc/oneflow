import numpy as np
import oneflow as flow


def test_eager_tensor_list_input(test_case):
    flow.clear_default_session()
    flow.enable_eager_execution()

    input_0 = np.random.rand(1, 5, 4).astype(np.single)
    input_1 = np.random.rand(1, 4, 4).astype(np.single)

    @flow.global_function()
    def foo_job(
        input_def=flow.MirroredTensorListDef(shape=(2, 5, 4), dtype=flow.float)
    ):
        output_0, output_1 = flow.tensor_list_split(input_def)
        test_case.assertTrue(np.array_equal(output_0.numpy(), input_0.squeeze()))
        test_case.assertTrue(np.array_equal(output_1.numpy(), input_1.squeeze()))

    foo_job([[input_0, input_1]])


def test_tensor_list_input_output(test_case):
    flow.clear_default_session()

    input_0 = np.random.rand(1, 5, 4).astype(np.single)
    input_1 = np.random.rand(1, 4, 4).astype(np.single)

    @flow.global_function()
    def foo_job(
        input_def=flow.MirroredTensorListDef(shape=(2, 5, 4), dtype=flow.float)
    ):
        tensor_buffer = flow.tensor_list_to_tensor_buffer(input_def)
        return flow.tensor_buffer_to_tensor_list(
            tensor_buffer, shape=(5, 4), dtype=flow.float
        )

    ret = foo_job([[input_0, input_1]]).get()
    ndarray_lists = ret.ndarray_lists()
    assert isinstance(ndarray_lists, (list, tuple))
    assert len(ndarray_lists) == 1
    ndarray_list = ndarray_lists[0]
    assert isinstance(ndarray_list, (list, tuple))
    assert len(ndarray_list) == 2
    test_case.assertTrue(np.array_equal(ndarray_list[0], input_0))
    test_case.assertTrue(np.array_equal(ndarray_list[1], input_1))

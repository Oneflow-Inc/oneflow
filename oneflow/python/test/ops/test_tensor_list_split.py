import numpy as np
import oneflow as flow


def _gen_random_input_list(input_static_shape):
    assert len(input_static_shape) > 1
    input_list = []
    for i in range(input_static_shape[0]):
        input_list.append(
            np.random.rand(
                1,
                np.random.randint(low=1, high=input_static_shape[1]),
                *input_static_shape[2:]
            ).astype(np.single)
        )
    return input_list


def _of_tensor_list_split(input_tensor_list, input_static_shape, device_tag="gpu"):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.device_prior_placement(device_tag, "0:0"))

    @flow.global_function(func_config)
    def tensor_list_split_job(
        input_def=flow.MirroredTensorListDef(
            shape=tuple(input_static_shape), dtype=flow.float
        ),
    ):
        outputs = flow.tensor_list_split(input_def)
        return outputs

    outputs = tensor_list_split_job([input_tensor_list]).get()
    return [output.ndarray_list()[0] for output in outputs]


def test_tensor_list_input_output(test_case, verbose=False):
    input_shape = [2, 5, 4]
    input_list = _gen_random_input_list(input_shape)
    output_list = _of_tensor_list_split(input_list, input_shape)
    for input, output in zip(input_list, output_list):
        if verbose:
            print("=" * 20)
            print(type(input))
            print("input:", input.shape, "\n", input.squeeze(0))
            print("output:", output.shape, "\n", output)
        test_case.assertTrue(np.array_equal(input.squeeze(0), output))

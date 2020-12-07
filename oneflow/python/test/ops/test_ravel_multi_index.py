import oneflow as flow
import numpy as np
import oneflow.typing as tp
from test_util import GenArgList
import unittest
from collections import OrderedDict
from typing import Dict
import os


def _compare_ravel_with_np(input, dims, device_type, machine_ids, device_counts):

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    def np_ravel(input, dims):
        return np.ravel_multi_index(input, dims)

    np_out_ravel = np_ravel(input, dims)

    # Convert Tuple to numpy.array for job_function
    of_dims = np.array(dims).astype(np.int32)
    @flow.global_function(
        type="predict", function_config=func_config,
    )
    def oneflow_ravel(
        of_input_1: tp.Numpy.Placeholder(shape=input[0].shape, dtype=flow.int32), 
        of_input_2: tp.Numpy.Placeholder(shape=input[1].shape, dtype=flow.int32),
        of_dims: tp.Numpy.Placeholder(shape=of_dims.shape, dtype=flow.int32)
    ) -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            of_ravel_out = flow.ravel_multi_index([of_input_1, of_input_2], of_dims)

        return of_ravel_out

    of_out_ravel = oneflow_ravel(input[0], input[1], of_dims)
    assert np.allclose(of_out_ravel, np_out_ravel)


def _gen_arg_dict(input, dims, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input"] = [input]
    arg_dict["dims"] = [dims]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Testravel1n1d(flow.unittest.TestCase):
    def test_ravel_cpu(test_case):
        a = np.array([3, 6, 2]).astype(np.int32)
        b = np.array([4, 5, 1]).astype(np.int32)
        c = np.stack([a, b], axis=0)
        dims = (8, 8)
        arg_dict = _gen_arg_dict(
            input=c, dims=dims, device_type="cpu", machine_ids="0:0", device_counts=1
        )
        for arg in GenArgList(arg_dict):
            _compare_ravel_with_np(*arg)

# TODO: Add test case for GPU

#     @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
#     def test_swish_gpu(test_case):
#         arg_dict = _gen_arg_dict(
#             shape=(3, 16, 32),
#             beta=10,
#             device_type="gpu",
#             machine_ids="0:0",
#             device_counts=1,
#         )
#         for arg in GenArgList(arg_dict):
#             _compare_swish_with_np(*arg)


# @flow.unittest.skip_unless_1n2d()
# class Teststack1n2d(flow.unittest.TestCase):
#     @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
#     def test_swish_gpu_1n2d(test_case):
#         arg_dict = _gen_arg_dict(
#             shape=(3, 8, 8, 4),
#             beta=2,
#             device_type="gpu",
#             machine_ids="0:0-1",
#             device_counts=2,
#         )
#         for arg in GenArgList(arg_dict):
#             _compare_swish_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList, type_name_to_flow_type

import oneflow as flow


def _test_tensor_buffer_convert(test_case, device):
    input = flow.Tensor(
        np.random.rand(16, 24, 32, 36), dtype=flow.float32, device=flow.device(device)
    )
    tensor_buffer = flow.tensor_to_tensor_buffer(input, instance_dims=2)
    orig_tensor = flow.tensor_buffer_to_tensor(
        tensor_buffer, dtype=flow.float32, instance_shape=[32, 36]
    )
    test_case.assertTrue(np.array_equal(input.numpy(), orig_tensor.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestTensorBufferOps(flow.unittest.TestCase):
    def test_tensor_buffer_convert(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_tensor_buffer_convert]
        arg_dict["device"] = ["cpu"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

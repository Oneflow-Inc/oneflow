import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow as flow


def _test_type_as(test_case, shape, device, src_dtype, tgt_dtype):
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=src_dtype, device=device)
    target = flow.tensor(np_input, dtype=tgt_dtype, device=device)
    input = input.type_as(target)
    test_case.assertEqual(input.dtype, target.dtype)


def _test_long(test_case, shape, device, dtype):
    np_input = np.random.rand(*shape)
    input = flow.tensor(np_input, dtype=dtype, device=device)
    input = input.long()
    test_case.assertEqual(input.dtype, flow.int64)


@flow.unittest.skip_unless_1n1d()
class TestTensorOps(flow.unittest.TestCase):
    def test_type_as(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["src_dtype"] = [flow.int64, flow.int32, flow.float32, flow.float64]
        arg_dict["tgt_dtype"] = [flow.int64, flow.int32, flow.float32, flow.float64]
        for arg in GenArgList(arg_dict):
            _test_type_as(test_case, *arg)

    def test_long(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2), (3, 4, 5), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [flow.int64, flow.int32, flow.float32, flow.float64]
        for arg in GenArgList(arg_dict):
            _test_long(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

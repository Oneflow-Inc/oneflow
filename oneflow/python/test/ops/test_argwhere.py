"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList
import oneflow.typing as oft


def _np_dtype_to_of_dtype(np_dtype):
    if np_dtype == np.float32:
        return flow.float
    elif np_dtype == np.int32:
        return flow.int32
    elif np_dtype == np.int64:
        return flow.int64
    else:
        raise NotImplementedError


def _random_input(shape, dtype):
    if dtype == np.float32:
        rand_ = np.random.random_sample(shape).astype(np.float32)
        rand_[np.nonzero(rand_ < 0.5)] = 0.0
        return rand_
    elif dtype == np.int32:
        return np.random.randint(low=0, high=2, size=shape).astype(np.int32)
    else:
        raise NotImplementedError


def _of_argwhere(x, index_dtype, device_type="gpu", dynamic=False):
    data_type = _np_dtype_to_of_dtype(x.dtype)
    out_data_type = _np_dtype_to_of_dtype(index_dtype)

    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(data_type)

    def do_argwhere(x_blob):
        with flow.scope.placement(device_type, "0:0"):
            return flow.argwhere(x_blob, dtype=out_data_type)

    if dynamic is True:
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def argwhere_fn(x_def: oft.ListNumpy.Placeholder(x.shape, dtype=data_type)):
            return do_argwhere(x_def)

        return argwhere_fn([x]).get().numpy_list()[0]

    else:
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def argwhere_fn(x_def: oft.Numpy.Placeholder(x.shape, dtype=data_type)):
            return do_argwhere(x_def)

        return argwhere_fn(x).get().numpy_list()[0]


def _compare_with_np(
    test_case, shape, value_dtype, index_dtype, device_type, dynamic, verbose=False
):
    x = _random_input(shape, value_dtype)
    y = np.argwhere(x)
    of_y = _of_argwhere(x, index_dtype, device_type, dynamic)
    if verbose is True:
        print("input:", x)
        print("np result:", y)
        print("of result:", of_y)
    test_case.assertTrue(np.array_equal(y, of_y))


@flow.unittest.skip_unless_1n1d()
class TestArgwhere(flow.unittest.TestCase):
    def test_argwhere(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(10), (30, 4), (8, 256, 20)]
        arg_dict["value_dtype"] = [np.float32, np.int32]
        arg_dict["index_dtype"] = [np.int32, np.int64]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["dynamic"] = [True, False]
        arg_dict["verbose"] = [False]
        for arg in GenArgList(arg_dict):
            _compare_with_np(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

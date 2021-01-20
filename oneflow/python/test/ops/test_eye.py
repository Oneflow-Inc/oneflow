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
import oneflow as flow
import numpy as np
import oneflow.typing as tp
from test_util import GenArgList
import unittest
from collections import OrderedDict
from typing import Dict
import os


def _compare_ones_with_np(n, m, device_type, machine_ids, device_counts):
    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()

    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_eye() -> tp.Numpy:
        of_eye = flow.eye(n, m)
        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(of_eye)
        return of_eye

    np_out_eye = np.eye(n, m, dtype=np.float32)

    of_out_eye = oneflow_eye()

    assert np.allclose(of_out_eye, np_out_eye)


def _gen_arg_dict(n, device_type, machine_ids, device_counts, m=None):
    # Generate a dict to pass parameter to test case
    if m is None:
        m = n
    arg_dict = OrderedDict()
    arg_dict["n"] = [n]
    arg_dict["m"] = [m]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Testones1n1d(flow.unittest.TestCase):
    def test_ones_cpu(test_case):
        arg_dict = _gen_arg_dict(
            n=3, device_type="cpu", machine_ids="0:0", device_counts=1
        )
        for arg in GenArgList(arg_dict):
            _compare_ones_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_ones_gpu(test_case):
        arg_dict = _gen_arg_dict(
            n=3, m=4, device_type="gpu", machine_ids="0:0", device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_ones_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testones1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_ones_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            n=5, m=6, device_type="gpu", machine_ids="0:0-1", device_counts=2,
        )
        for arg in GenArgList(arg_dict):
            _compare_ones_with_np(*arg)


if __name__ == "__main__":
    unittest.main()

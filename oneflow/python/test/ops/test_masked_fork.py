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
import oneflow.typing as oft
from typing import Tuple
from test_util import GenArgList
import unittest
from collections import OrderedDict
import os


def gen_masked_fork_test_sample(input_shape, is_float=True):
    def _np_masked_fork(input, masked):
        out_true = np.zeros(input_shape)
        out_false = np.zeros(input_shape)
        for i in range(0, input.size):
            coor = np.unravel_index(i, input.shape)
            val = input[coor]
            if masked[np.unravel_index(i, masked.shape)]:
                out_true[coor] = val
            else:
                out_false[coor] = val
        return out_true, out_false

    if is_float:
        input = np.random.random(input_shape)
    else:
        input = np.random.randint(0, 100, input_shape)
    mask = np.random.randint(0, 2, input_shape)
    out_true, out_false = _np_masked_fork(input, mask)
    ret = {"input": input, "mask": mask, "out_true": out_true, "out_false": out_false}
    return ret


def _make_masked_fork_fn(
    test_case,
    input,
    mask,
    device_type,
    value_type,
    mask_type,
    machine_ids,
    device_counts,
):
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    # global function needs float32 as type of argument and return value
    if value_type == flow.float16:
        func_config.default_data_type(flow.float32)
    else:
        func_config.default_data_type(value_type)

    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))
    func_config.default_logical_view(flow.scope.consistent_view())

    if (
        value_type == flow.float32
        or value_type == flow.float64
        or value_type == flow.int32
        or value_type == flow.int8
        or value_type == flow.int64
    ):

        @flow.global_function(type="predict", function_config=func_config)
        def masked_fork_fn(
            input: oft.Numpy.Placeholder(input.shape, dtype=value_type),
            mask: oft.Numpy.Placeholder(input.shape, dtype=mask_type),
        ) -> Tuple[oft.Numpy, oft.Numpy]:
            with flow.scope.placement(device_type, "0:0"):
                in_var = flow.get_variable(
                    "input",
                    shape=input.shape,
                    dtype=value_type,
                    initializer=flow.constant_initializer(0),
                )
                in_var = flow.cast_to_current_logical_view(in_var)
                x = in_var + input

            out_ture, out_false = flow.masked_fork(x, mask)
            return out_ture, out_false

        return masked_fork_fn
    else:
        raise Exception("value type unimplemented")


def _compare_masked_fork_with_samples(
    test_case, device_type, sample, value_type, mask_type, machine_ids, device_count
):
    masked_fork_fn = _make_masked_fork_fn(
        test_case,
        sample["input"].astype(value_type[0]),
        sample["mask"].astype(mask_type[0]),
        device_type,
        value_type[1],
        mask_type[1],
        machine_ids,
        device_count,
    )
    out_true, out_false = masked_fork_fn(
        sample["input"].astype(value_type[0]), sample["mask"].astype(mask_type[0])
    )
    out_true.astype(value_type[0])
    out_false.astype(value_type[0])

    if value_type == flow.float16:
        raise "float not supported yet"
    else:
        test_case.assertTrue(
            np.allclose(out_true, sample["out_true"].astype(value_type[0]))
        )
        test_case.assertTrue(
            np.allclose(out_false, sample["out_false"].astype(value_type[0]))
        )


def _gen_arg_dict(device_type="gpu", machine_ids="0:0", device_count=1):
    arg_dict = OrderedDict()
    arg_dict["device_type"] = [device_type]
    arg_dict["samples"] = []
    arg_dict["samples"].append(gen_masked_fork_test_sample((2, 2)))
    arg_dict["samples"].append(gen_masked_fork_test_sample((4, 3)))
    arg_dict["value_type"] = [(np.float64, flow.float64), (np.float32, flow.float32)]
    arg_dict["mask_type"] = [(np.int32, flow.int32)]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_count"] = [device_count]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class TestMaskedFork1n1d(flow.unittest.TestCase):
    def test_masked_fork_float_cpu(test_case):
        arg_dict = _gen_arg_dict("cpu", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_masked_fork_with_samples(test_case, *arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_masked_fork_float_gpu(test_case):
        arg_dict = _gen_arg_dict("gpu", "0:0", 1)
        for arg in GenArgList(arg_dict):
            _compare_masked_fork_with_samples(test_case, *arg)


@flow.unittest.skip_unless_1n2d()
class TestMaskedFork1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_masked_fork_float(test_case):
        arg_dict = _gen_arg_dict("gpu", "0:0-1", 2)
        for arg in GenArgList(arg_dict):
            _compare_masked_fork_with_samples(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

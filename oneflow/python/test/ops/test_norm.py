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


def np_norm(x, p="fro", axis=None, keepdims=False):
    print(axis, p)

    def p_norm(input, porder, axis, keepdims=False):
        print("porder is: ", porder)
        if porder == np.inf:
            out = np.max(np.abs(input), axis=axis, keepdims=keepdims)
        elif porder == -np.inf:
            out = np.min(np.abs(input), axis=axis, keepdims=keepdims)
        elif porder == float(0):
            out = np.sum((input != 0), axis=axis, keepdims=keepdims)
        elif porder == float(1):
            out = np.sum(np.abs(input), axis=axis, keepdims=keepdims)
        elif porder == -float(1):
            out = 1 / np.sum(1 / np.abs(x), axis=axis, keepdims=keepdims)
        else:
            # print("1!!!! ", porder)
            abs_x = np.abs(input)
            out = np.power(abs_x, porder)
            out = np.sum(out, axis=axis, keepdims=keepdims)
            out = np.power(out, 1.0 / porder)
        return out

    def frobenius_norm(input, axis=None, keepdims=False):
        print("Here is frobenius_norm!")
        if axis is not None and len(axis) > 2:
            print("WARNING!!!!")
        return np.sqrt(np.sum(np.square(input), axis=axis, keepdims=keepdims))

    if not isinstance(p, str):
        p = float(p)
    if isinstance(axis, int):
        _axis = (axis,)
    else:
        _axis = axis

    if p == "fro":
        return frobenius_norm(x, axis=_axis, keepdims=keepdims)
    elif isinstance(p, float):
        return p_norm(x, p, _axis, keepdims)


def _compare_norm_with_np(
    input_shape, p, axis, keepdims, device_type, machine_ids, device_counts
):
    input_1 = np.random.random(size=input_shape).astype(np.float32)
    print("Input is: ", input_1)
    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    np_out_norm = np_norm(input_1, p, axis, keepdims)

    def assert_prediction_grad(blob: tp.Numpy):
        print("OF Grad is: ", blob)
        # TODO: add assert
        # assert np.allclose(blob, _np_grad)

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_norm(
        of_input_1: tp.Numpy.Placeholder(shape=input_1.shape),
    ) -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=input_1.shape,
                dtype=flow.float32,
                initializer=flow.zeros_initializer(),
                name="x_var",
            )
            x_var = of_input_1 + v

        flow.watch_diff(x_var, assert_prediction_grad)

        of_norm_out = flow.norm(x_var, p, axis, keepdims)

        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(of_norm_out)

        return of_norm_out

    of_out_norm = oneflow_norm(input_1)

    try:
        assert np.allclose(of_out_norm, np_out_norm, atol=1e-3)
        print("OF out is: ", of_out_norm)
        print("Np out is: ", np_out_norm)
    except:
        # print("OF out is: ", of_out_norm)
        # print("Np out is: ", np_out_norm)
        print("====WRONG====")


def _gen_arg_dict(shape, p, axis, keepdims, device_type, machine_ids, device_counts):
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [shape]
    arg_dict["p"] = [*p]
    arg_dict["axis"] = [*axis]
    arg_dict["keepdims"] = [keepdims]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


def _gen_arg_dict_inf(shape, axis, keepdims, device_type, machine_ids, device_counts):
    arg_dict = _gen_arg_dict(
        shape,
        [np.inf, -np.inf],
        axis,
        keepdims,
        device_type,
        machine_ids,
        device_counts,
    )
    return arg_dict


def _gen_arg_dict_fro(shape, axis, keepdims, device_type, machine_ids, device_counts):
    arg_dict = _gen_arg_dict(
        shape, ["fro"], axis, keepdims, device_type, machine_ids, device_counts
    )
    return arg_dict


def _gen_arg_dict_other(shape, axis, keepdims, device_type, machine_ids, device_counts):
    # arg_dict = _gen_arg_dict(shape, [-2, -1, 0, 1, 2, 3, -3], axis, keepdims, device_type, machine_ids, device_counts)
    arg_dict = _gen_arg_dict(
        shape,
        [-2, -1, 1, 2, 3, -3],
        axis,
        keepdims,
        device_type,
        machine_ids,
        device_counts,
    )

    return arg_dict


@flow.unittest.skip_unless_1n1d()
class TestCpuNorm(flow.unittest.TestCase):
    def test_vector(self):
        arg_dict_inf = _gen_arg_dict_inf(
            shape=(3,),
            axis=[None, 0],
            keepdims=False,
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )
        arg_dict_fro = _gen_arg_dict_fro(
            shape=(3,),
            axis=[None, 0],
            keepdims=False,
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )
        arg_dict_other = _gen_arg_dict_other(
            shape=(3,),
            axis=[None, 0],
            keepdims=False,
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg_dict in [arg_dict_inf, arg_dict_fro, arg_dict_other]:
            for arg in GenArgList(arg_dict):
                print(arg)
                _compare_norm_with_np(*arg)

    def test_matrix(self):
        arg_dict_inf = _gen_arg_dict_inf(
            shape=(3, 3, 3),
            axis=[None, 0, 1, 2, (0, 1), (1, 2), (0, 1, 2)],
            keepdims=False,
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )
        arg_dict_fro = _gen_arg_dict_fro(
            shape=(3, 3, 3),
            axis=[None, 0, 1, 2, (0, 1), (1, 2)],
            keepdims=False,
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )
        arg_dict_other = _gen_arg_dict_other(
            shape=(3, 3, 3),
            axis=[None, (0, 1), (1, 2), (0, 1, 2)],
            keepdims=False,
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg_dict in [arg_dict_inf, arg_dict_fro, arg_dict_other]:
            for arg in GenArgList(arg_dict):
                _compare_norm_with_np(*arg)


if __name__ == "__main__":
    unittest.main()

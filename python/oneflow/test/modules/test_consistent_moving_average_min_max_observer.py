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
import numpy as np
import oneflow as flow
import oneflow.unittest

from collections import OrderedDict
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
from test_moving_average_min_max_observer import (
    gen_quant_scale_for_moving_average_min_max_symmetric,
)
from test_moving_average_min_max_observer import (
    gen_quant_scale_for_moving_average_min_max_affine,
)
from test_moving_average_min_max_observer import (
    gen_quant_scale_for_moving_average_min_max_cambricon,
)
from test_moving_average_min_max_observer import _check_moving_average_min_max_observer
from oneflow.test_utils.automated_test_util import *


def _run_test_moving_average_min_max_observer(
    test_case,
    device_type,
    dtype,
    activation_shape,
    quantization_bit,
    quantization_scheme,
    quantization_formula,
    momentum,
    placement,
):
    moving_max_np = np.zeros((1,))
    moving_min_np = np.zeros((1,))
    for sbp in all_sbp(placement, max_dim=1):
        if any([item == flow.sbp.split(0) for item in sbp]):
            continue
        current_train_step_tensor = flow.tensor(
            np.zeros((1,)).astype(np.float32), dtype=flow.int64
        ).to_consistent(placement, sbp)
        for i in range(10):
            activation = (np.random.random(activation_shape) - 0.5).astype(
                type_name_to_np_type[dtype]
            )
            activation_tensor = flow.tensor(
                activation, dtype=flow.float32
            ).to_consistent(placement, sbp)
            moving_average_min_max_observer = flow.nn.MovingAverageMinMaxObserver(
                training=True,
                quantization_formula=quantization_formula,
                stop_update_after_iters=1,
                quantization_bit=quantization_bit,
                quantization_scheme=quantization_scheme,
                momentum=momentum,
            )
            moving_average_min_max_observer = moving_average_min_max_observer.to_consistent(
                placement, sbp
            )
            (scale, zero_point) = moving_average_min_max_observer(
                activation_tensor, current_train_step_tensor
            )
            _check_moving_average_min_max_observer(
                test_case,
                activation,
                scale.numpy(),
                zero_point.numpy(),
                moving_max_np,
                moving_min_np,
                quantization_bit,
                quantization_scheme,
                quantization_formula,
                momentum,
            )


class TestMovingAverageMinMaxObserver(flow.unittest.TestCase):
    @flow.unittest("wrong result when 1n2d")
    @consistent
    def test_moving_average_min_max_observer(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["cpu", "cuda"]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["activation_shape"] = [(9, 40, 20, 10)]
        arg_dict["quantization_bit"] = [8, 2]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["momentum"] = [0.95]
        arg_dict["placement"] = list(all_placement())
        for arg in GenArgList(arg_dict):
            _run_test_moving_average_min_max_observer(*arg)


if __name__ == "__main__":
    unittest.main()

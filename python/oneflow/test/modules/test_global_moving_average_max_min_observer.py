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
from collections import OrderedDict
from oneflow.test_utils.test_util import GenArgList
from test_moving_average_min_max_observer import _check_moving_average_min_max_observer
from oneflow.test_utils.automated_test_util import *


def _run_test_moving_average_min_max_observer(
    test_case,
    placement,
    sbp,
    device_type,
    dtype,
    activation_shape,
    quantization_bit,
    quantization_scheme,
    quantization_formula,
    momentum,
):
    moving_max_np = np.zeros((1,))
    moving_min_np = np.zeros((1,))

    current_train_step_tensor = flow.tensor(
        np.zeros((1,)).astype(np.float32),
        dtype=flow.int64,
        placement=placement,
        sbp=sbp,
    )
    for i in range(10):
        of_activation = (
            random_tensor(len(activation_shape), *activation_shape, low=-0.5, high=0.5)
            .to_global(placement, sbp)
            .oneflow
        )
        np_activation = of_activation.numpy()

        moving_average_min_max_observer = flow.nn.MovingAverageMinMaxObserver(
            quantization_formula=quantization_formula,
            stop_update_after_iters=1,
            quantization_bit=quantization_bit,
            quantization_scheme=quantization_scheme,
            momentum=momentum,
        )
        moving_average_min_max_observer = moving_average_min_max_observer.to_global(
            placement, sbp
        )
        (scale, zero_point) = moving_average_min_max_observer(
            of_activation, current_train_step_tensor
        )
        _check_moving_average_min_max_observer(
            test_case,
            np_activation,
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
    @globaltest
    def test_moving_average_min_max_observer(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "cuda"]
        arg_dict["dtype"] = ["float32", "double"]
        arg_dict["activation_shape"] = [(9, 48, 24, 10)]
        arg_dict["quantization_bit"] = [8, 2]
        arg_dict["quantization_scheme"] = ["symmetric", "affine"]
        arg_dict["quantization_formula"] = ["google"]
        arg_dict["momentum"] = [0.95]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, valid_split_axis=[1, 2]):
                    _run_test_moving_average_min_max_observer(
                        test_case, placement, sbp, *arg
                    )


if __name__ == "__main__":
    unittest.main()

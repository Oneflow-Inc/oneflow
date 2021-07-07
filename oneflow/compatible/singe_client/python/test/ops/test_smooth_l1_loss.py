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
import uuid
from collections import OrderedDict

import os
import numpy as np
import oneflow as flow
import oneflow.typing as oft
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


def gen_numpy_data(prediction, label, beta=1.0):
    original_shape = prediction.shape
    elem_cnt = prediction.size
    prediction = prediction.reshape(-1)
    label = label.reshape(-1)
    loss = np.zeros((elem_cnt)).astype(prediction.dtype)
    prediction_grad = np.zeros((elem_cnt)).astype(prediction.dtype)

    # Forward
    for i in np.arange(elem_cnt):
        abs_diff = abs(prediction[i] - label[i])
        if abs_diff < beta:
            loss[i] = 0.5 * abs_diff * abs_diff / beta
        else:
            loss[i] = abs_diff - 0.5 * beta

    # Backward
    for i in np.arange(elem_cnt):
        diff = prediction[i] - label[i]
        abs_diff = abs(diff)
        if abs_diff < beta:
            prediction_grad[i] = diff / beta
        else:
            prediction_grad[i] = np.sign(diff)

    return {
        "loss": loss.reshape(original_shape),
        "prediction_grad": prediction_grad.reshape(original_shape),
    }


@flow.unittest.skip_unless_1n1d()
class TestSmoothL1Loss(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_smooth_l1_loss(_):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["prediction_shape"] = [
            (100,),
            (10, 10),
        ]
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["beta"] = [0, 0.5, 1]

        for case in GenArgList(arg_dict):
            device_type, prediction_shape, data_type, beta = case
            assert device_type in ["gpu", "cpu"]
            assert data_type in ["float32", "double", "int8", "int32", "int64"]
            flow.clear_default_session()
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)

            prediction = np.random.randn(*prediction_shape).astype(
                type_name_to_np_type[data_type]
            )
            label = np.random.randn(*prediction_shape).astype(
                type_name_to_np_type[data_type]
            )

            np_result = gen_numpy_data(prediction, label, beta)

            def assert_prediction_grad(b):
                prediction_grad = np_result["prediction_grad"]
                assert prediction_grad.dtype == type_name_to_np_type[data_type]
                assert np.allclose(prediction_grad, b.numpy()), (
                    case,
                    prediction_grad,
                    b.numpy(),
                )

            @flow.global_function(type="train", function_config=func_config)
            def TestJob(
                prediction: oft.Numpy.Placeholder(
                    prediction_shape, dtype=type_name_to_flow_type[data_type]
                ),
                label: oft.Numpy.Placeholder(
                    prediction_shape, dtype=type_name_to_flow_type[data_type]
                ),
            ):
                v = flow.get_variable(
                    "prediction",
                    shape=prediction_shape,
                    dtype=type_name_to_flow_type[data_type],
                    initializer=flow.constant_initializer(0),
                    trainable=True,
                )
                flow.watch_diff(v, assert_prediction_grad)
                prediction += v
                with flow.scope.placement(device_type, "0:0"):
                    loss = flow.smooth_l1_loss(prediction, label, beta)
                    flow.optimizer.SGD(
                        flow.optimizer.PiecewiseConstantScheduler([], [1e-4]),
                        momentum=0,
                    ).minimize(loss)
                    return loss

            loss_np = np_result["loss"]
            assert loss_np.dtype == type_name_to_np_type[data_type]
            loss = TestJob(prediction, label).get().numpy()
            assert np.allclose(loss_np, loss), (case, loss_np, loss)


if __name__ == "__main__":
    unittest.main()

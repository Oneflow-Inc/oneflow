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
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft


def dynamic_loss_scale_schedule(
    count_not_finite, loss_scale, good_step_counter, increment_period, multiplier, name,
):
    (
        flow.user_op_builder(name)
        .Op("dynamic_loss_scale_schedule")
        .Input("count_not_finite", [count_not_finite])
        .Input("loss_scale", [loss_scale])
        .Input("good_step_counter", [good_step_counter])
        .Attr("increment_period", increment_period)
        .Attr("multiplier", multiplier)
        .Build()
        .InferAndTryRun()
    )


def _run_test(
    test_case, device_type, op_param,
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def schedule_job(count_not_finite: oft.Numpy.Placeholder((1,), dtype=flow.int64),):
        with flow.scope.placement(device_type, "0:0"):
            good_step_counter = flow.get_variable(
                name="good_step_counter",
                shape=(1,),
                dtype=flow.int64,
                initializer=flow.constant_initializer(
                    op_param["good_step_counter_value"], dtype=flow.int64
                ),
            )
            loss_scale = flow.get_variable(
                name="loss_scale",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.constant_initializer(
                    op_param["loss_scale_value"], dtype=flow.float
                ),
            )
            dynamic_loss_scale_schedule(
                count_not_finite,
                loss_scale,
                good_step_counter,
                op_param["increment_period"],
                op_param["multiplier"],
                "dynamic_schedule",
            )
            return good_step_counter, loss_scale

    @flow.global_function(function_config=func_config)
    def fetch_job():
        with flow.scope.placement(device_type, "0:0"):
            good_step_counter = flow.get_variable(
                name="good_step_counter",
                shape=(1,),
                dtype=flow.int64,
                initializer=flow.constant_initializer(
                    op_param["good_step_counter_value"], dtype=flow.int64
                ),
            )
            loss_scale = flow.get_variable(
                name="loss_scale",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.constant_initializer(
                    op_param["loss_scale_value"], dtype=flow.float
                ),
            )
        return good_step_counter, loss_scale

    count_not_finite = np.array([op_param["count_not_finite"]]).astype(np.int64)
    schedule_job(count_not_finite).get()
    good_step_counter, loss_scale = fetch_job().get()

    assert good_step_counter.numpy()[0] == op_param["result_step"]
    assert loss_scale.numpy()[0] == op_param["result_loss_scale"]


@flow.unittest.skip_unless_1n1d()
class TestDynamicLossScaleSchedule(flow.unittest.TestCase):
    def test_dynamic_loss_scale_schedule(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_case"] = [test_case]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["op_param"] = [
            {
                "count_not_finite": 1,
                "good_step_counter_value": 1,
                "loss_scale_value": 100.0,
                "increment_period": 1,
                "multiplier": 2.0,
                "result_step": 0,
                "result_loss_scale": 50.0,
            },
            {
                "count_not_finite": 0,
                "good_step_counter_value": 1,
                "loss_scale_value": 100.0,
                "increment_period": 1,
                "multiplier": 2.0,
                "result_step": 0,
                "result_loss_scale": 200.0,
            },
            {
                "count_not_finite": 0,
                "good_step_counter_value": 1,
                "loss_scale_value": 100.0,
                "increment_period": 10,
                "multiplier": 2.0,
                "result_step": 2,
                "result_loss_scale": 100.0,
            },
        ]
        for arg in GenArgList(arg_dict):
            _run_test(*arg)


if __name__ == "__main__":
    unittest.main()

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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import test_global_storage
from test_util import Args, GenArgDict, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft
import unittest


def margin_loss(loss_m1, loss_m2, loss_m3, s, inputs, labels):
    inputs = inputs * s
    class_num = inputs.shape[1]
    if loss_m1 != 1.0 or loss_m2 != 0.0 or loss_m3 != 0.0:
        if loss_m1 == 1.0 and loss_m2 == 0.0:
            s_m = s * loss_m3
            gt_one_hot = flow.one_hot(
                labels, depth=class_num, on_value=s_m, off_value=0.0, dtype=flow.float,
            )
            inputs = inputs - gt_one_hot
        else:
            labels_expand = flow.reshape(labels, (labels.shape[0], 1))
            zy = flow.gather(inputs, labels_expand, batch_dims=1)
            cos_t = zy * (1 / s)
            t = flow.math.acos(cos_t)
            if loss_m1 != 1.0:
                t = t * loss_m1
            if loss_m2 > 0.0:
                t = t + loss_m2
            body = flow.math.cos(t)
            if loss_m3 > 0.0:
                body = body - loss_m3
            new_zy = body * s
            diff = new_zy - zy
            gt_one_hot = flow.one_hot(
                labels, depth=class_num, on_value=1.0, off_value=0.0, dtype=flow.float,
            )
            body = gt_one_hot * diff
            inputs = inputs + body
    return inputs


def test_combined_margin_loss(
    test_case, device_type, input_shape, label_shape, data_type, m1, m2, m3, s
):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def test_job(
        x: oft.Numpy.Placeholder(input_shape, dtype=flow.float32),
        labels: oft.Numpy.Placeholder(label_shape, dtype=flow.int32),
    ):
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                name="v",
                shape=(1,),
                dtype=flow.float32,
                initializer=flow.zeros_initializer(),
            )
            x = x + v

            x1 = flow.identity(x)
            x2 = flow.identity(x)

            flow.watch_diff(x1, test_global_storage.Setter("x1_diff"))
            flow.watch_diff(x2, test_global_storage.Setter("x2_diff"))

            x1 = flow.cast(x1, data_type)
            x2 = flow.cast(x2, data_type)

        with flow.scope.placement(device_type, "0:0-3"):
            y1 = (
                flow.combined_margin_loss(
                    x1.with_distribute(flow.distribute.split(1)),
                    labels.with_distribute(flow.distribute.broadcast()),
                    m1,
                    m2,
                    m3,
                )
                * s
            )
            y2 = margin_loss(m1, m2, m3, s, x2, labels)

        with flow.scope.placement(device_type, "0:0"):
            y1 = flow.cast(y1, flow.float)
            y2 = flow.cast(y2, flow.float)

            flow.watch(y1, test_global_storage.Setter("y1"))
            flow.watch(y2, test_global_storage.Setter("y2"))
            loss = y1 + y2
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0.001]), momentum=0
            ).minimize(flow.math.reduce_sum(loss))

        return loss

    x = np.random.uniform(low=-1, high=1, size=input_shape).astype(np.float32)
    labels = np.random.randint(0, 1000, size=(*label_shape,)).astype(np.int32)
    test_job(x, labels).get()

    tol = 2e-3

    y1 = test_global_storage.Get("y1")
    y2 = test_global_storage.Get("y2")

    test_case.assertTrue(np.allclose(y1, y2, rtol=tol, atol=tol))
    x1_diff = test_global_storage.Get("x1_diff")
    x2_diff = test_global_storage.Get("x2_diff")
    test_case.assertTrue(np.allclose(x1_diff, x2_diff, rtol=tol, atol=tol))


@flow.unittest.skip_unless_1n4d()
class TestCombinedMarginLoss(flow.unittest.TestCase):
    def test_case(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["input_shape"] = [(64, 1000)]
        arg_dict["label_shape"] = [(64,)]
        arg_dict["data_type"] = [flow.float32]
        arg_dict["m1"] = [0.3]
        arg_dict["m2"] = [0.5]
        arg_dict["m3"] = [0.4]
        arg_dict["s"] = [5]
        for arg in GenArgDict(arg_dict):
            test_combined_margin_loss(test_case, **arg)


if __name__ == "__main__":
    unittest.main()

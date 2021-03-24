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
import numpy as np
import oneflow as flow
import oneflow.typing as oft
import unittest
import os
from collections import OrderedDict
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import test_global_storage


def distributed_gather(params, indices, name):
    return (
        flow.user_op_builder(name)
        .Op("distributed_gather")
        .Input("in", [params])
        .Input("indices", [indices])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def _run_test(test_case, device_type, x_shape, indices_shape):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def FlowJob(
        x: oft.Numpy.Placeholder(x_shape, dtype=flow.float),
        indices: oft.Numpy.Placeholder(indices_shape, dtype=flow.int32),
    ):
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=flow.float,
                initializer=flow.constant_initializer(0),
                trainable=True,
            )
            x += v
        with flow.scope.placement(device_type, "0:0-3"):
            loss = distributed_gather(x, indices, name="distributed_gather")

        with flow.scope.placement(device_type, "0:0"):
            loss = flow.identity(loss)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)
            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(loss, test_global_storage.Setter("loss"))
            flow.watch_diff(loss, test_global_storage.Setter("loss_diff"))
        return loss

    x = np.random.randn(*x_shape).astype(np.float32)
    indices = np.random.randint(0, x_shape[0], size=indices_shape).astype(np.int32)
    my_loss = FlowJob(x, indices).get()
    diff = my_loss.numpy() - x[indices]
    assert diff.sum() == 0
    loss_diff = test_global_storage.Get("loss_diff")
    x_diff = test_global_storage.Get("x_diff")

    np_x_diff = np.zeros(x_shape)
    for i in range(indices.size):
        np_x_diff[indices[i]] += loss_diff[i]

    diff = np_x_diff - x_diff
    assert diff.sum() == 0


@flow.unittest.skip_unless_1n4d()
class TestDistributedGather(flow.unittest.TestCase):
    def test_distributed_gather(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(100, 3)]
        arg_dict["indices_shape"] = [
            (60,),
        ]
        for arg in GenArgList(arg_dict):
            _run_test(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

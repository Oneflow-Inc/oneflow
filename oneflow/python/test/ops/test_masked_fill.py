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
from collections import OrderedDict
import numpy as np
import oneflow as flow
import torch

from test_util import (
    GenArgDict,
    test_global_storage,
    type_name_to_flow_type,
    type_name_to_np_type,
)
import oneflow.typing as oft


def _test_masked_fill_fw_bw(test_case, device, x_shape, mask_shape, type_name, value=0):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    np_type = type_name_to_np_type[type_name]
    flow_type = type_name_to_flow_type[type_name]

    func_config.default_data_type(flow_type)

    @flow.global_function(type="train", function_config=func_config)
    def test_masked_fill_fw_bw_job(
        x: oft.Numpy.Placeholder(x_shape, dtype=flow_type),
        mask: oft.Numpy.Placeholder(mask_shape, dtype=flow_type),
    ):
        with flow.scope.placement(device, "0:0"):
            y = flow.get_variable(
                name="vx",
                shape=(1,),
                dtype=flow.float,
                initializer=flow.zeros_initializer(),
            )
            y = flow.cast(y, dtype=flow_type)
            x += y
            mask = flow.cast(mask, dtype=flow_type)
            out = flow.math.masked_fill(x, mask, value)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(out)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(out, test_global_storage.Setter("out"))
            flow.watch_diff(out, test_global_storage.Setter("out_diff"))
            return out

    check_point = flow.train.CheckPoint()
    check_point.init()
    x = np.random.randint(low=0, high=100, size=x_shape)
    mask = np.random.randint(low=0, high=1, size=mask_shape)

    test_masked_fill_fw_bw_job(x.astype(np_type), mask.astype(np_type)).get()
    out_diff = test_global_storage.Get("out_diff")

    torch_x = torch.Tensor(x).float()
    torch_x.requires_grad = True
    torch_mask = torch.ByteTensor(mask)
    touch_y = torch_x.masked_fill(torch_mask, value)
    touch_y.backward(torch.Tensor(out_diff))

    test_case.assertTrue(
        np.all(
            touch_y.detach().numpy().astype(np_type) == test_global_storage.Get("out")
        )
    )

    test_case.assertTrue(
        np.all(
            torch_x.grad.numpy().astype(np_type) == test_global_storage.Get("x_diff")
        )
    )


def test_masked_fill_fw_bw(test_case):
    arg_dict = OrderedDict()
    arg_dict["type_name"] = ["float32", "double", "int8", "int32", "int64"]
    arg_dict["device"] = ["gpu", "cpu"]
    arg_dict["x_shape"] = [
        (2, 4),
        (1, 4),
        (2, 3, 4),
        (2, 1, 4),
        (2, 3, 3, 4),
        (4, 2, 3, 4, 4),
    ]
    arg_dict["mask_shape"] = [(2, 1, 1, 4)]
    arg_dict["value"] = [2.5, 3.3, -5.5]
    for arg in GenArgDict(arg_dict):
        _test_masked_fill_fw_bw(test_case, **arg)

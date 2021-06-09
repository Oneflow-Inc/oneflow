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

import oneflow
import numpy as np

import oneflow.experimental as flow
from test_util import GenArgList


def _test_instruction_replay_impl(test_case, device, shape):
    x = flow.Tensor(np.random.rand(*shape), device=flow.device(device))
    y = flow.Tensor(np.random.rand(*shape), device=flow.device(device))

    x.determine()
    y.determine()

    oneflow._oneflow_internal.start_recording_instruction()
    z = x + y
    oneflow._oneflow_internal.end_recording_instruction()

    test_case.assertTrue(np.allclose(z.numpy(), x.numpy() + y.numpy(), 1e-4, 1e-4))

    # init tensor_z and replay
    z.zeros_()
    oneflow._oneflow_internal.replay_instruction()
    test_case.assertTrue(np.allclose(z.numpy(), x.numpy() + y.numpy(), 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestIntructionReplay(flow.unittest.TestCase):
    def test_instruction_replay(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [[2, 3], [1, 10]]
        for arg in GenArgList(arg_dict):
            _test_instruction_replay_impl(test_case, *arg)


if __name__ == "__main__":
    unittest.main()

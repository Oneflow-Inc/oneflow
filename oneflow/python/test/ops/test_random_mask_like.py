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
import os
import shutil
import tempfile
from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList, type_name_to_flow_type
import oneflow.typing as oft


def of_run(device_type, x_shape, rate, seed):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    @flow.global_function(function_config=func_config)
    def RandomMaskLikeJob(x: oft.Numpy.Placeholder(x_shape)):
        with flow.scope.placement(device_type, "0:0"):
            mask = flow.nn.random_mask_like(x, rate=rate, seed=seed, name="random_mask")
            return mask

    # OneFlow
    x = np.random.rand(*x_shape).astype(np.float32)
    of_out = RandomMaskLikeJob(x).get().numpy()
    assert np.allclose(
        [1 - np.count_nonzero(of_out) / of_out.size], [rate], atol=rate / 5
    )


@flow.unittest.skip_unless_1n1d()
class TestRandomMaskLike(flow.unittest.TestCase):
    def test_random_mask_like(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["x_shape"] = [(100, 100, 10, 20), (100, 100, 200)]
        arg_dict["rate"] = [0.1, 0.4, 0.75]
        arg_dict["seed"] = [12345, None]

        for arg in GenArgList(arg_dict):
            of_run(*arg)


if __name__ == "__main__":
    unittest.main()

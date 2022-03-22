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
from collections import OrderedDict

from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
from oneflow.test_utils.automated_test_util import *


def _test_fused_scale_tril(
    test_case, shape, diagonal, fill_value, scale, placement, sbp
):
    x = random_tensor(len(shape), *shape).oneflow
    global_x = x.to_global(placement=placement, sbp=sbp)
    global_x.retain_grad()
    global_y = flow._C.fused_scale_tril(global_x, diagonal, fill_value, scale)
    global_y.sum().backward()

    local_x = x.to_local().to(placement.type)
    local_x.retain_grad()
    local_y = flow._C.fused_scale_tril(local_x, diagonal, fill_value, scale)
    local_y.sum().backward()

    test_case.assertTrue(np.allclose(global_y.numpy(), local_y.numpy()))
    test_case.assertTrue(np.allclose(global_x.grad.numpy(), local_x.grad.numpy()))


class FusedScaleTrilTestCase(flow.unittest.TestCase):
    @globaltest
    def test_fused_scale_tril(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(8, 8), (16, 32)]
        arg_dict["diagonal"] = [-1, 0, 1]
        arg_dict["fill_value"] = [-1, 0, 1]
        arg_dict["scale"] = [-2.3, 0.7, 2]
        for kwargs in GenArgList(arg_dict):
            for placement in all_placement():
                if placement.type != "cuda":
                    continue
                for sbp in all_sbp(placement, max_dim=2):
                    _test_fused_scale_tril(test_case, *kwargs, placement, sbp)


if __name__ == "__main__":
    unittest.main()

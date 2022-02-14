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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *

def check_equality(x):
    equality_res = np.allclose(
        x.pytorch.detach().cpu().numpy(),
        x.oneflow.numpy(),
        rtol=0.0001,
        atol=1e-05,
        equal_nan=True,
    )
    if not equality_res:
        print(x.oneflow.sbp)
        print("Pytorch: ", x.pytorch.detach().cpu().numpy())
        print("Oneflow: ", x.oneflow.numpy())
    return equality_res

def test_flip_impl(test_case, ndim, placement, sbp):
    dims = [4, 4]
    x = random_tensor(ndim, *dims)
    y = x.to_global(placement=placement, sbp=sbp)
    z = torch.flip(y, [0])
    assert check_equality(z), "z is not equal"

class TestFlipConsistent(flow.unittest.TestCase):
    @global_view
    def test_flip(test_case):
        ndim = 2
        placement = flow.env.all_device_placement("cpu")
        sbp = (flow.sbp.split(axis=0),)
        test_flip_impl(test_case, ndim, placement, sbp)

if __name__ == "__main__":
    unittest.main()

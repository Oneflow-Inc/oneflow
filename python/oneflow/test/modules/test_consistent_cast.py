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
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


data_types = [
    (flow.float, np.float32),
    # (flow.float16, np.float16),
    (flow.double, np.double),
    (flow.int, np.int32),
    (flow.int64, np.int64),
    (flow.int8, np.int8),
    # (flow.uint8, np.uint8),
]


def _test_cast(test_case, ndim, ori_dtype_index, dst_dtype_index, placement, sbp):
    dims = [random(1, 3).to(int).value() * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims).to_global(placement, sbp).oneflow
    x.retain_grad()
    np_x = x.numpy()

    flow_ori_dtype = data_types[ori_dtype_index][0]
    np_ori_dtype = data_types[ori_dtype_index][1]
    flow_dst_dtype = data_types[dst_dtype_index][0]
    np_dst_dtype = data_types[dst_dtype_index][1]

    y = flow.cast(x, flow_ori_dtype)
    y = flow.cast(y, flow_dst_dtype)
    np_y = np_x.astype(np_ori_dtype)
    np_y = np_y.astype(np_dst_dtype)
    test_case.assertTrue(np.array_equal(np_y, y.numpy()))

    if y.requires_grad:
        y.sum().backward()
        test_case.assertTrue(np.array_equal(x.grad.numpy(), np.ones(dims)))


class TestCast(flow.unittest.TestCase):
    @globaltest
    def test_cast(test_case):
        for placement in all_placement():
            ndim = random(1, 4).to(int).value()
            for sbp in all_sbp(placement, max_dim=min(2, ndim)):
                ori_dtype_index = random(0, len(data_types)).to(int).value()
                dst_dtype_index = random(0, len(data_types)).to(int).value()

                # float16 is not supported for cpu device
                if (
                    ori_dtype_index == 1 or dst_dtype_index == 1
                ) and placement.type != "cuda":
                    continue
                _test_cast(
                    test_case, ndim, ori_dtype_index, dst_dtype_index, placement, sbp
                )


if __name__ == "__main__":
    unittest.main()

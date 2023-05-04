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
import oneflow as flow
import oneflow.unittest
import numpy as np


def _test_index_select(test_case, mlu_placement, cpu_placement, x_sbp, index_sbp):
    shape = (32, 8, 24, 12)
    dim = 0
    x_shape = shape
    if x_sbp == flow.sbp.split(0):
        x_shape = (int(shape[0] / 2), shape[1], shape[2], shape[3])

    local_x = flow.tensor(np.random.randn(*x_shape), device="cpu", dtype=flow.float32)

    local_index = flow.tensor(
        np.random.randint(low=0, high=x_shape[dim], size=20),
        device="cpu",
        dtype=flow.int32,
    )

    cpu_x = local_x.to_global(cpu_placement, x_sbp)
    cpu_index = local_index.to_global(cpu_placement, index_sbp)

    x = local_x.to_global(mlu_placement, x_sbp)
    index = local_index.to_global(mlu_placement, index_sbp)

    cpu_out = flow.index_select(cpu_x, dim, cpu_index)
    mlu_out = flow.index_select(x, dim, index)

    test_case.assertTrue(np.allclose(cpu_x.numpy(), x.numpy(), 1e-4, 1e-4))
    test_case.assertTrue(np.allclose(cpu_index.numpy(), index.numpy(), 1e-4, 1e-4))
    test_case.assertTrue(np.allclose(cpu_out.numpy(), mlu_out.numpy(), 1e-4, 1e-4))


@flow.unittest.skip_unless_1n2d()
class TestIndexSelectModule(flow.unittest.TestCase):
    def test_index_select(test_case):
        mlu_placement = flow.placement("mlu", [0, 1])
        cpu_placement = flow.placement("cpu", [0, 1])
        all_sbp = (flow.sbp.broadcast, flow.sbp.split(0), flow.sbp.partial_sum)
        index_sbps = (flow.sbp.broadcast, flow.sbp.split(0))
        for x_sbp in all_sbp:
            for index_sbp in index_sbps:
                _test_index_select(
                    test_case, mlu_placement, cpu_placement, x_sbp, index_sbp
                )


if __name__ == "__main__":
    unittest.main()

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


def _test_add(test_case, mlu_placement, cpu_placement, shape, x_sbp, y_sbp):
    x_shape = shape
    y_shape = shape
    if x_sbp == flow.sbp.split(0):
        x_shape = (int(shape[0] / 2), shape[1])
    if y_sbp == flow.sbp.split(0):
        y_shape = (int(shape[0] / 2), shape[1])

    x_arry = np.random.randn(*x_shape).astype(float)
    y_arry = np.random.randn(*y_shape).astype(float)

    a = flow.tensor(x_arry, device=mlu_placement.type, dtype=flow.float)
    b = flow.tensor(y_arry, device=mlu_placement.type, dtype=flow.float)

    cpu_a = flow.tensor(x_arry, device=cpu_placement.type, dtype=flow.float)
    cpu_b = flow.tensor(y_arry, device=cpu_placement.type, dtype=flow.float)

    x = a.to_global(mlu_placement, x_sbp)
    y = b.to_global(mlu_placement, y_sbp)

    cpu_x = cpu_a.to_global(cpu_placement, x_sbp)
    cpu_y = cpu_b.to_global(cpu_placement, y_sbp)

    z = flow.add(x, y)
    cpu_z = flow.add(cpu_x, cpu_y)

    test_case.assertTrue(np.allclose(z.numpy(), cpu_z.numpy(), 0.0001, 0.0001))


@flow.unittest.skip_unless_1n2d()
class TestAddModule(flow.unittest.TestCase):
    def test_add(test_case):
        mlu_placement = flow.placement("mlu", [0, 1])
        cpu_placement = flow.placement("cpu", [0, 1])
        shape = (4, 768)
        all_sbp = (flow.sbp.broadcast, flow.sbp.split(0), flow.sbp.partial_sum)
        for x_sbp in all_sbp:
            for y_sbp in all_sbp:
                _test_add(test_case, mlu_placement, cpu_placement, shape, x_sbp, y_sbp)


if __name__ == "__main__":
    unittest.main()

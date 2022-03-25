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
from test_in_top_k import _in_top_k_np


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_in_top_k_impl(test_case, placement, sbp):
    dims = [random(1, 3).to(int) * 8 for _ in range(2)]
    x_np = (
        random_pytorch_tensor(1, dims[0], high=dims[1], dtype=int)
        .value()
        .detach()
        .cpu()
        .numpy()
    )
    y_np = random_pytorch_tensor(2, *dims).value().detach().cpu().numpy()
    x = flow.tensor(x_np, dtype=flow.int32).to_global(placement=placement, sbp=sbp)
    y = flow.tensor(y_np, dtype=flow.float32).to_global(placement=placement, sbp=sbp)
    of_out = flow.in_top_k(x, y, 2)
    np_out = _in_top_k_np(x_np, y_np, 2)
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


class TestInTopK(flow.unittest.TestCase):
    @globaltest
    def test_in_top_k(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement):
                _test_in_top_k_impl(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()

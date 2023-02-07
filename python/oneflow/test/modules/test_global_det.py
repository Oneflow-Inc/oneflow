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

import re
import unittest
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


def det_all_placement():
    cuda_version = flow._oneflow_internal.flags.cuda_version()
    if cuda_version < 11000:  # cuSOLVER is only supported in CUDA 11.0 and above
        return all_cpu_placement()
    else:
        # FIXME: remove this after fixing the bug of cuda global det
        return all_cpu_placement()
        # return all_placement()


@autotest(n=1, check_graph=False, auto_backward="auto")
def _test_det(test_case, placement, sbp, ndim):
    dim_list = [random(1, 3).to(int).value() * 8 for _ in range(ndim - 2)]
    square_dim = 8
    dim_list.extend([square_dim] * 2)
    x = (
        random_tensor(ndim, *dim_list, low=-1)
        .to(torch.double)
        .to_global(placement, sbp)
    )
    return torch.linalg.det(x)


class TestDet(flow.unittest.TestCase):
    @globaltest
    def test_det(test_case):
        ndim = random(2, 5).to(int).value()
        for placement in det_all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_det(test_case, placement, sbp, ndim)


if __name__ == "__main__":
    unittest.main()

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
from oneflow.test_utils.automated_test_util.util import broadcast


def _test_batch_gather(test_case, ndim, placement, sbp):
    dims = [random(1, 3).to(int).value() * 8 for _ in range(ndim)]
    x = random_tensor(ndim, *dims, requires_grad=True)
    local_x = flow.tensor(x.pytorch.detach().cpu().numpy(), requires_grad=True)
    global_x = x.oneflow.to_global(placement=placement, sbp=sbp)
    global_x.retain_grad()

    indices_ndim = random(1, ndim + 1).to(int).value()
    indices_dims = [dims[i] for i in range(indices_ndim)]
    indices_dims[-1] = random(1, dims[indices_ndim - 1]).to(int).value()
    indices = np.random.choice(dims[indices_ndim - 1], indices_dims)
    indices = broadcast(indices)
    local_indices = flow.tensor(indices)
    global_indices = local_indices.to_global(
        placement=placement, sbp=[flow.sbp.broadcast for _ in range(len(sbp))]
    )

    global_out = flow.batch_gather(global_x, global_indices)
    global_out.sum().backward()
    local_out = flow.batch_gather(local_x, local_indices)
    local_out.sum().backward()
    test_case.assertTrue(
        np.allclose(
            global_x.grad.detach().cpu().numpy(),
            local_x.grad.detach().cpu().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )
    )


class TestBatchGather(flow.unittest.TestCase):
    @globaltest
    def test_batch_gather(test_case):
        ndim = 2
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=ndim):
                _test_batch_gather(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()

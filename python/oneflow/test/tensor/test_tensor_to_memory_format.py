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
import random as random_util

import oneflow as flow
import oneflow.unittest
import numpy as np

from oneflow.test_utils.automated_test_util import *


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestTensor(flow.unittest.TestCase):
    @autotest(n=3)
    def test_to_memory_format(test_case):
        def check_equal(a, b):
            test_case.assertEqual(list(a.shape), list(b.shape))
            test_case.assertEqual(list(a.stride()), list(b.stride()))
            test_case.assertEqual(a.is_contiguous(), b.is_contiguous())
            test_case.assertTrue(
                np.allclose(
                    a.detach().cpu().numpy(), b.detach().cpu().numpy(), 1e-06, 1e-06
                )
            )

        device = random_device()
        x = random_tensor(
            ndim=4,
            dim0=random(1, 6).to(int),
            dim1=random(1, 6).to(int),
            dim2=random(1, 6).to(int),
            dim3=random(1, 6).to(int),
        ).to(device)

        oneflow_x = x.oneflow
        pytorch_x = x.pytorch

        # TODO(): implement backward
        with flow.no_grad():
            oneflow_y = oneflow_x.to(memory_format=torch.contiguous_format.oneflow)
            pytorch_y = pytorch_x.to(memory_format=torch.contiguous_format.pytorch)
            check_equal(oneflow_y, pytorch_y)

            oneflow_y = oneflow_x.to(memory_format=torch.channels_last.oneflow)
            pytorch_y = pytorch_x.to(memory_format=torch.channels_last.pytorch)
            # Note: pytorch Tensor.to(channels_last) won't change tensor shape, so we should
            #       permute it that only change the tensor shape and won't relayout its storage.
            # TODO(): align with pytorch
            check_equal(oneflow_y, pytorch_y.permute(0, 2, 3, 1))


if __name__ == "__main__":
    unittest.main()

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

from oneflow.test_utils.automated_test_util import *
import oneflow as flow
import oneflow.unittest
import numpy as np


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestPartialFC(flow.unittest.TestCase):
    def test_partial_fc(test_case):
        p = flow.env.all_device_placement("cuda")
        w = flow.randn(
            50000, 128, placement=p, sbp=flow.sbp.broadcast, requires_grad=True
        )
        label = flow.randint(0, 50000, (512,), placement=p, sbp=flow.sbp.broadcast)
        num_sample = 5000
        out = flow.distributed_partial_fc_sample(w, label, num_sample)
        test_case.assertTrue(out[0].shape == flow.Size([512]))
        test_case.assertTrue(out[1].shape == flow.Size([5000]))
        test_case.assertTrue(out[2].shape == flow.Size([5000, 128]))

        # test gradient function
        x = flow.randn(
            512, 128, placement=p, sbp=flow.sbp.broadcast, requires_grad=True
        )
        _, sample_labels, sample_weights = out
        flow.matmul(x, sample_weights, transpose_b=True).sum().backward()

        # test numerical correctness
        sample_mask = flow.BoolTensor(w.shape[0], placement=p, sbp=flow.sbp.broadcast)
        sample_mask[:] = False
        sample_mask[sample_labels] = True

        test_case.assertTrue(
            np.allclose(
                w.grad[sample_labels].numpy(),
                np.repeat(flow.sum(x, dim=0, keepdim=True).numpy(), num_sample, axis=0),
                rtol=0.01,
                atol=0.01,
            )
        )
        test_case.assertTrue(
            np.allclose(
                w.grad[~sample_mask].numpy(),
                np.zeros((w.shape[0] - num_sample, w.shape[1])),
                rtol=0.0001,
                atol=0.0001,
            )
        )
        test_case.assertTrue(
            np.allclose(
                x.grad.numpy(),
                np.repeat(
                    flow.sum(w[sample_labels], dim=0, keepdim=True).numpy(),
                    x.shape[0],
                    axis=0,
                ),
                rtol=0.01,
                atol=0.01,
            )
        )


if __name__ == "__main__":
    unittest.main()

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

import oneflow.experimental as flow
from automated_test_util import *


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMaskedFill(flow.unittest.TestCase):
    def test_masked_fill_aginst_pytorch(test_case):
        import numpy as np
        import torch

        def mask_tensor(shape):
            def generator(_):
                rng = np.random.default_rng()
                np_arr = rng.integers(low=0, high=2, size=shape)
                return (
                    flow.Tensor(np_arr, dtype=flow.int8),
                    torch.tensor(np_arr, dtype=torch.bool),
                )

            return generator

        for device in ["cpu", "cuda"]:
            test_flow_against_pytorch(
                test_case,
                "masked_fill",
                extra_annotations={"mask": flow.Tensor, "value": float},
                extra_generators={
                    "input": random_tensor(ndim=2, dim0=4, dim1=5),
                    "mask": mask_tensor((4, 5)),
                    "value": constant(3.14),
                },
                device=device,
            )

            test_tensor_against_pytorch(
                test_case,
                "masked_fill",
                extra_annotations={"mask": flow.Tensor, "value": float},
                extra_generators={
                    "input": random_tensor(ndim=2, dim0=4, dim1=5),
                    "mask": mask_tensor((4, 5)),
                    "value": constant(3.14),
                },
                device=device,
            )


if __name__ == "__main__":
    unittest.main()

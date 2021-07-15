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
import numpy as np
import unittest

import oneflow.experimental as flow

from automated_test_util import *


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestConvTranspose2d(flow.unittest.TestCase):
    def test_with_random_data(test_case):
        for device in ["cpu", "cuda"]:
            spatial_size = np.random.randint(32, 64)
            in_channel = np.random.randint(1, 129)
            test_module_against_pytorch(
                test_case,
                "nn.ConvTranspose2d",
                extra_annotations={"device": None, "dtype": None},
                extra_generators={
                    "input": random_tensor(
                        ndim=4, dim1=in_channel, dim2=spatial_size, dim3=spatial_size
                    ),
                    "in_channels": constant(in_channel),
                    "out_channels": random(1, 129),
                    # NOTE(Liang Depeng):
                    # When kernel_size = 1, the results of oneflow are not the same with torch.
                    # But this problem can be handled later, because usually
                    # no one will use nn.ConvTranspose2d by setting kernel_size to 1.
                    "kernel_size": random(2, 4),
                    "stride": random(1, 4),
                    "padding": random(1, 5),
                    "dilation": random(1, 3),
                    "groups": random(1, 5),
                    "output_padding": constant(0),
                    "padding_mode": constant("zeros"),
                    "bias": constant(False),
                    "device": constant("float32"),
                    "dtype": constant(device),
                },
                device=device,
                rtol=1e-2,
            )


if __name__ == "__main__":
    unittest.main()

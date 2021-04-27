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
import oneflow as flow
import unittest
import numpy as np
from oneflow.python.nn.modules.utils import (
    _single,
    _pair,
    _triple,
    _reverse_repeat_tuple,
)

g_samples = [
    {
        "kernel": (3, 2),
        "padding": 0,
        "stride": (2, 1),
        "in": np.array(
            [
                [
                    [
                        [
                            -0.1953,
                            1.3992,
                            -0.7464,
                            0.6910,
                            -1.5484,
                            0.4970,
                            1.4963,
                            0.3080,
                            -1.4730,
                            -0.1238,
                        ],
                        [
                            -0.3532,
                            1.2078,
                            -0.3796,
                            0.7326,
                            -1.5795,
                            0.2128,
                            0.6501,
                            -0.1266,
                            -1.3121,
                            0.1483,
                        ],
                        [
                            -0.3412,
                            -1.6446,
                            -1.0039,
                            -0.5594,
                            0.7450,
                            -0.5323,
                            -1.6887,
                            0.2399,
                            1.9422,
                            0.4214,
                        ],
                        [
                            -1.6362,
                            -1.2234,
                            -1.2531,
                            0.6109,
                            0.2228,
                            -0.2080,
                            0.6359,
                            0.2451,
                            0.3864,
                            0.4263,
                        ],
                        [
                            0.7053,
                            0.3413,
                            0.9090,
                            -0.4057,
                            -0.2830,
                            1.0444,
                            -0.2884,
                            0.7638,
                            -1.4793,
                            0.2079,
                        ],
                        [
                            -0.1207,
                            0.8458,
                            -0.9521,
                            0.3630,
                            0.1772,
                            0.3945,
                            0.4056,
                            -0.7822,
                            0.6166,
                            1.3343,
                        ],
                        [
                            -0.4115,
                            0.5802,
                            1.2909,
                            1.6508,
                            -0.0561,
                            -0.7964,
                            0.9786,
                            0.4265,
                            0.7262,
                            0.2819,
                        ],
                        [
                            -0.2667,
                            -0.0792,
                            0.4771,
                            0.3248,
                            -0.1313,
                            -0.3325,
                            -0.9973,
                            0.3128,
                            -0.5151,
                            -0.1225,
                        ],
                        [
                            -1.4983,
                            0.2604,
                            -0.9127,
                            0.0822,
                            0.3708,
                            -2.6024,
                            0.2249,
                            -0.7500,
                            0.3152,
                            0.1931,
                        ],
                        [
                            -0.2171,
                            -0.2602,
                            0.9051,
                            -0.0933,
                            -0.0902,
                            -1.3837,
                            -1.2519,
                            -1.3091,
                            0.7155,
                            2.3376,
                        ],
                    ]
                ]
            ]
        ),
        "out": np.array(
            [
                [
                    [
                        [
                            0.0121,
                            -0.1946,
                            -0.2110,
                            -0.2531,
                            -0.3675,
                            0.1059,
                            0.1465,
                            -0.0703,
                            -0.0662,
                        ],
                        [
                            -0.6331,
                            -0.6458,
                            -0.2837,
                            0.0551,
                            0.1648,
                            -0.1729,
                            -0.0154,
                            0.3497,
                            0.3175,
                        ],
                        [
                            0.3234,
                            0.5025,
                            0.4760,
                            0.2410,
                            0.0801,
                            0.2897,
                            0.2506,
                            0.0453,
                            0.2813,
                        ],
                        [
                            -0.2359,
                            0.2694,
                            0.4855,
                            0.3735,
                            -0.5913,
                            -0.5875,
                            0.0326,
                            0.0859,
                            0.1465,
                        ],
                    ]
                ]
            ]
        ),
        "ceil_mode": False,
    }
]


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestModule(flow.unittest.TestCase):
    def test_AvgPool2d(test_case):
        global g_samples
        for sample in g_samples:
            of_avgpool2d = flow.nn.AvgPool2d(
                kernel_size=sample["kernel"],
                padding=sample["padding"],
                stride=sample["stride"],
                ceil_mode=sample["ceil_mode"],
            )
            x = flow.Tensor(sample["in"])
            of_y = of_avgpool2d(x)
            test_case.assertTrue(of_y.numpy().shape == sample["out"].shape)
            test_case.assertTrue(np.allclose(of_y.numpy(), sample["out"], 1e-4, 1e-4))


if __name__ == "__main__":
    unittest.main()

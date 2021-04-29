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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestReLUModule(flow.unittest.TestCase):
    def test_relu(test_case):
        m = flow.nn.ReLU()
        arr = np.random.randn(2, 3, 4, 5)

        np_out = np.maximum(0, arr)
        x = flow.Tensor(arr)
        of_out = m(x)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestTanhModule(flow.unittest.TestCase):
    def _test_body_tanh(test_case, input_arr):
        x = flow.Tensor(input_arr)

        tanh = flow.nn.Tanh()
        y = tanh(x)
        z = np.tanh(input_arr)

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def _test_ones_body_tanh(self, shape):
        x = np.ones(shape, dtype=np.float32)
        self._test_body_tanh(x)

    def _test_random_body_tanh(self, shape):
        x = np.random.random(shape).astype(np.float32)
        self._test_body_tanh(x)

    def test_ones_input_tanh(self):
        self._test_ones_body_tanh((1))
        self._test_ones_body_tanh((1, 10))
        self._test_ones_body_tanh((2, 10, 2))
        self._test_ones_body_tanh((2, 5, 2, 2))

    def test_random_input_tanh(self):
        self._test_random_body_tanh((1))
        self._test_random_body_tanh((1, 10))
        self._test_random_body_tanh((2, 10, 2))
        self._test_random_body_tanh((2, 5, 2, 2))

    def _test_body_tanh_v2(test_case, input_arr):
        x = flow.Tensor(input_arr)

        y = flow.tanh(x)
        z = np.tanh(input_arr)

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def _test_body_tanh_v3(test_case, input_arr):
        x = flow.Tensor(input_arr)

        y = x.tanh()
        z = np.tanh(input_arr)

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestGeLU(flow.unittest.TestCase):
    def test_gelu_v1(test_case):
        input_arr = np.array([-0.5, 0, 0.5]).astype(np.float32)
        x = flow.Tensor(input_arr)

        gelu = flow.nn.GELU()
        y = gelu(x)
        z = np.array([-0.15426877, 0.0, 0.34573123])

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def test_gelu_v2(test_case):
        input_arr = np.array([-0.5, 0, 0.5]).astype(np.float32)
        x = flow.Tensor(input_arr)

        y = flow.gelu(x)
        z = np.array([-0.15426877, 0.0, 0.34573123])

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))

    def test_gelu_v3(test_case):
        input_arr = np.array([-0.5, 0, 0.5]).astype(np.float32)
        x = flow.Tensor(input_arr)

        y = x.gelu()

        z = np.array([-0.15426877, 0.0, 0.34573123])

        test_case.assertTrue(np.allclose(y.numpy(), z, rtol=1e-4, atol=1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSigmoidModule(flow.unittest.TestCase):
    def test_sigmoid(test_case):
        m = flow.nn.Sigmoid()
        x = flow.Tensor(
            np.array(
                [
                    [0.81733328, 0.43621480, 0.10351428],
                    [-1.15555191, -0.67776406, 0.27372134],
                ]
            )
        )
        y = m(x)
        y2 = flow.sigmoid(x)
        y3 = x.sigmoid()
        output = np.array(
            [[0.69366997, 0.60735673, 0.52585548], [0.23947647, 0.33676055, 0.56800622]]
        )
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))
        test_case.assertTrue(np.allclose(y2.numpy(), output, rtol=1e-05))
        test_case.assertTrue(np.allclose(y3.numpy(), output, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestSoftmaxModule(flow.unittest.TestCase):
    def test_softmax(test_case):
        m = flow.nn.Softmax(dim=1)
        x = flow.Tensor(
            np.array(
                [
                    [
                        [
                            [-0.46716809, 0.40112534, 0.61984003],
                            [-1.31244969, -0.42528763, 1.47953856],
                        ]
                    ],
                    [
                        [
                            [1.02978742, -0.49383053, 1.88214159],
                            [1.35351622, -1.46251285, -1.40751374],
                        ]
                    ],
                ]
            )
        )
        y = m(x)
        output = np.array(
            [[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]], [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]
        )
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

    def test_softmax_dim_2(test_case):
        x = flow.Tensor(
            np.array(
                [
                    [
                        [
                            [-0.46716809, 0.40112534, 0.61984003],
                            [-1.31244969, -0.42528763, 1.47953856],
                        ]
                    ],
                    [
                        [
                            [1.02978742, -0.49383053, 1.88214159],
                            [1.35351622, -1.46251285, -1.40751374],
                        ]
                    ],
                ]
            )
        )
        y = flow.softmax(x, dim=2)
        output = np.array(
            [
                [
                    [
                        [0.69957644, 0.69559592, 0.29740232],
                        [0.30042359, 0.30440408, 0.70259768],
                    ]
                ],
                [
                    [
                        [0.41976729, 0.72485679, 0.96407223],
                        [0.58023274, 0.27514324, 0.03592779],
                    ]
                ],
            ]
        )
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

    def test_softmax_dim_3(test_case):
        m = flow.nn.Softmax(dim=3)
        x = flow.Tensor(
            np.array(
                [
                    [
                        [
                            [-0.46716809, 0.40112534, 0.61984003],
                            [-1.31244969, -0.42528763, 1.47953856],
                        ]
                    ],
                    [
                        [
                            [1.02978742, -0.49383053, 1.88214159],
                            [1.35351622, -1.46251285, -1.40751374],
                        ]
                    ],
                ]
            )
        )
        y = m(x)
        output = np.array(
            [
                [
                    [
                        [0.15752424, 0.37535521, 0.46712062],
                        [0.05065432, 0.12300029, 0.82634538],
                    ]
                ],
                [
                    [
                        [0.28065580, 0.06116108, 0.65818310],
                        [0.89041674, 0.05328530, 0.05629803],
                    ]
                ],
            ]
        )
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLogSoftmaxModule(flow.unittest.TestCase):
    def test_logsoftmax(test_case):
        m = flow.nn.LogSoftmax(1)
        x = flow.Tensor(
            np.array([[0.4296, -1.1957, 2.5463], [1.2552, -1.5747, 0.6923]])
        )
        y = m(x)
        output = np.array(
            [
                [-2.25134873, -3.87664890, -0.13464880],
                [-0.48770463, -3.31760454, -1.05060458],
            ]
        )
        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))

    def test_logsoftmax_v2(test_case):
        m = flow.nn.LogSoftmax(dim=2)
        x = flow.Tensor(
            np.array(
                [
                    [
                        [
                            [2.55630851, 1.57471120, 0.25240266, -0.57634264],
                            [0.72222596, 0.35014620, 0.43715513, 1.41162395],
                            [0.12103304, -1.15901530, 1.08098269, 0.04042318],
                        ],
                        [
                            [-0.43854573, 1.07273626, 1.80571628, -1.72897887],
                            [1.33143651, -1.53906214, -1.45766914, -1.10325944],
                            [-0.14271064, 0.38371494, 0.45469931, -0.32640621],
                        ],
                    ],
                    [
                        [
                            [1.75221181, -1.04841065, -0.41832924, -0.56028533],
                            [0.89104170, -0.10217502, -0.21890916, -0.24784143],
                            [-0.35868922, -1.64597833, -0.19254614, 0.24511036],
                        ],
                        [
                            [0.08831860, 2.37159801, 0.79408669, -0.39868262],
                            [-0.69181246, -1.11924624, -0.47067565, 0.14795294],
                            [-0.23413812, -0.71034479, 0.44405901, -0.90119874],
                        ],
                    ],
                ]
            )
        )

        y = m(x)
        output = np.array(
            [
                [
                    [
                        [-0.22100820, -0.30664772, -1.50251734, -2.31782818],
                        [-2.05509090, -1.53121281, -1.31776488, -0.32986164],
                        [-2.65628386, -3.04037428, -0.67393732, -1.70106244],
                    ],
                    [
                        [-2.10596132, -0.45455331, -0.26023546, -1.93661511],
                        [-0.33597916, -3.06635165, -3.52362084, -1.31089568],
                        [-1.81012630, -1.14357471, -1.61125243, -0.53404248],
                    ],
                ],
                [
                    [
                        [-0.43424436, -1.41734290, -1.24530625, -1.52699995],
                        [-1.29541445, -0.47110727, -1.04588616, -1.21455598],
                        [-2.54514551, -2.01491070, -1.01952314, -0.72160423],
                    ],
                    [
                        [-0.78056860, -0.07357123, -0.68661338, -1.20370412],
                        [-1.56069970, -3.56441545, -1.95137572, -0.65706855],
                        [-1.10302532, -3.15551400, -1.03664112, -1.70622015],
                    ],
                ],
            ]
        )

        test_case.assertTrue(np.allclose(y.numpy(), output, rtol=1e-05))


if __name__ == "__main__":
    unittest.main()

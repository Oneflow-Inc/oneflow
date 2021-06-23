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
from collections import OrderedDict

import numpy as np

import oneflow.experimental as flow
from test_util import GenArgList


input_arr = np.array(
    [
        [
            [[-0.16046895, -1.03667831], [-0.34974465, 0.26505867]],
            [[-1.24111986, -0.53806001], [1.72426331, 0.43572459]],
        ],
        [
            [[-0.77390957, -0.42610624], [0.16398858, -1.35760343]],
            [[1.07541728, 0.11008703], [0.26361224, -0.48663723]],
        ],
    ],
    dtype=np.float32,
)


def _test_layernorm(test_case, device):
    output = np.array(
        [
            [
                [[-0.0544118, -1.0509688], [-0.2696846, 0.4295622]],
                [[-1.2834904, -0.4838651], [2.0891891, 0.6236691]],
            ],
            [
                [[-0.8555527, -0.3554582], [0.4930190, -1.6948260]],
                [[1.8035311, 0.4155158], [0.6362644, -0.4424936]],
            ],
        ],
        dtype=np.float32,
    )

    x = flow.Tensor(input_arr, device=flow.device(device))
    m = flow.nn.LayerNorm(x.size()[1:]).to(device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))


def _test_layernorm_v2(test_case, device):
    output = np.array(
        [
            [
                [[0.3406544, -1.5249983], [-0.0623574, 1.2467014]],
                [[-1.2004623, -0.5688803], [1.4634399, 0.3059027]],
            ],
            [
                [[-0.3180245, 0.3122248], [1.3815271, -1.3757277]],
                [[1.4972910, -0.2341234], [0.0412391, -1.3044068]],
            ],
        ],
        dtype=np.float32,
    )
    x = flow.Tensor(input_arr, device=flow.device(device))
    m = flow.nn.LayerNorm([2, 2], eps=1e-5).to(device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))


def _test_layernorm_v3(test_case, device):
    output = np.array(
        [
            [
                [[0.9999740, -0.9999740], [-0.9999470, 0.9999470]],
                [[-0.9999595, 0.9999595], [0.9999880, -0.9999880]],
            ],
            [
                [[-0.9998344, 0.9998341], [0.9999914, -0.9999914]],
                [[0.9999787, -0.9999787], [0.9999645, -0.9999645]],
            ],
        ],
        dtype=np.float32,
    )
    x = flow.Tensor(input_arr, device=flow.device(device))
    m = flow.nn.LayerNorm(2, elementwise_affine=True).to(device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output, 1e-05, 1e-05))


def _test_layernorm_backward(test_case, device):
    output = np.array(
        [
            [
                [[-0.0544118, -1.0509688], [-0.2696846, 0.4295622]],
                [[-1.2834904, -0.4838651], [2.0891891, 0.6236691]],
            ],
            [
                [[-0.8555527, -0.3554582], [0.4930190, -1.6948260]],
                [[1.8035311, 0.4155158], [0.6362644, -0.4424936]],
            ],
        ],
        dtype=np.float32,
    )

    x = flow.Tensor(input_arr, device=flow.device(device), requires_grad=True)
    m = flow.nn.LayerNorm(x.size()[1:]).to(device=flow.device(device))
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.zeros(shape=input_arr.shape), 1e-5, 1e-5)
    )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestLayerNorm(flow.unittest.TestCase):
    def test_layernorm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_layernorm,
            _test_layernorm_v2,
            _test_layernorm_v3,
            _test_layernorm_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

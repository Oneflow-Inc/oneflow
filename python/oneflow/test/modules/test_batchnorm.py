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
from automated_test_util import *
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_batchnorm1d_2d_input(test_case, device):
    input_arr = np.array(
        [
            [0.1438, 1.1229, -0.048, -1.6834, -0.8262],
            [0.5836, 0.135, -0.886, -1.7878, 1.0592],
            [0.7252, -1.1488, -0.0274, 1.4051, 0.1018],
            [-0.3595, -0.1801, 0.1146, -1.5712, -1.9291],
        ],
        dtype=np.float32,
    )
    output_arr = np.array(
        [
            [-0.3056, 1.4066, 0.4151, -0.5783, -0.3864],
            [0.7326, 0.1884, -1.71, -0.6563, 1.317],
            [1.0668, -1.3949, 0.4674, 1.7292, 0.4521],
            [-1.4938, -0.2002, 0.8275, -0.4945, -1.3827],
        ],
        dtype=np.float32,
    )
    m = flow.nn.BatchNorm1d(num_features=5, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output_arr, rtol=0.0001, atol=0.0001))


def _test_batchnorm1d_3d_input(test_case, device):
    input_arr = np.array(
        [
            [
                [-0.1091, 2.0041, 0.885, -0.0412],
                [-1.2055, 0.7442, 2.33, 1.2411],
                [-1.2466, 0.3667, 1.2267, 0.3043],
            ],
            [
                [-0.2484, -1.1407, 0.3352, 0.6687],
                [-0.2975, -0.0227, -0.2302, -0.3762],
                [-0.7759, -0.6789, 1.1444, 1.8077],
            ],
        ],
        dtype=np.float32,
    )
    output_arr = np.array(
        [
            [
                [-0.464, 1.9673, 0.6798, -0.3859],
                [-1.4207, 0.4529, 1.9767, 0.9303],
                [-1.4831, 0.096, 0.9379, 0.035],
            ],
            [
                [-0.6243, -1.651, 0.0471, 0.4309],
                [-0.5481, -0.284, -0.4834, -0.6237],
                [-1.0224, -0.9274, 0.8573, 1.5066],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.BatchNorm1d(num_features=3, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output_arr, rtol=0.0001, atol=0.0001))


def _test_batchnorm2d(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [-0.8791, 0.2553, 0.7403, -0.2859],
                    [0.8006, -1.7701, -0.9617, 0.1705],
                    [0.2842, 1.7825, 0.3365, -0.8525],
                ],
                [
                    [0.7332, -0.0737, 0.7245, -0.6551],
                    [1.4461, -0.1827, 0.9737, -2.1571],
                    [0.4657, 0.7244, 0.3378, 0.1775],
                ],
            ],
            [
                [
                    [1.8896, 1.8686, 0.1896, 0.9817],
                    [-0.0671, 1.5569, 1.1449, 0.0086],
                    [-0.9468, -0.0124, 1.3227, -0.6567],
                ],
                [
                    [-0.8472, 1.3012, -1.1065, 0.9348],
                    [1.0346, 1.5703, 0.2419, -0.7048],
                    [0.6957, -0.4523, -0.8819, 1.0164],
                ],
            ],
        ],
        dtype=np.float32,
    )
    output = np.array(
        [
            [
                [
                    [-1.1868, -0.0328, 0.4606, -0.5833],
                    [0.522, -2.0933, -1.2709, -0.119],
                    [-0.0034, 1.5209, 0.0498, -1.1598],
                ],
                [
                    [0.5601, -0.3231, 0.5505, -0.9595],
                    [1.3404, -0.4424, 0.8233, -2.6035],
                    [0.2673, 0.5504, 0.1273, -0.0482],
                ],
            ],
            [
                [
                    [1.6299, 1.6085, -0.0996, 0.7062],
                    [-0.3608, 1.2914, 0.8723, -0.2837],
                    [-1.2557, -0.3051, 1.0531, -0.9606],
                ],
                [
                    [-1.1698, 1.1818, -1.4536, 0.7807],
                    [0.89, 1.4763, 0.0223, -1.0139],
                    [0.519, -0.7375, -1.2078, 0.87],
                ],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.BatchNorm2d(num_features=2, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.Tensor(input_arr, device=flow.device(device), dtype=flow.float32)
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output, 0.0001, 0.0001))


def _test_batchnorm2d_track_running_stats(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [-0.8791, 0.2553, 0.7403, -0.2859],
                    [0.8006, -1.7701, -0.9617, 0.1705],
                    [0.2842, 1.7825, 0.3365, -0.8525],
                ],
                [
                    [0.7332, -0.0737, 0.7245, -0.6551],
                    [1.4461, -0.1827, 0.9737, -2.1571],
                    [0.4657, 0.7244, 0.3378, 0.1775],
                ],
            ],
            [
                [
                    [1.8896, 1.8686, 0.1896, 0.9817],
                    [-0.0671, 1.5569, 1.1449, 0.0086],
                    [-0.9468, -0.0124, 1.3227, -0.6567],
                ],
                [
                    [-0.8472, 1.3012, -1.1065, 0.9348],
                    [1.0346, 1.5703, 0.2419, -0.7048],
                    [0.6957, -0.4523, -0.8819, 1.0164],
                ],
            ],
        ],
        dtype=np.float32,
    )
    output = np.array(
        [
            [
                [
                    [-1.1868, -0.0328, 0.4606, -0.5833],
                    [0.522, -2.0933, -1.2709, -0.119],
                    [-0.0034, 1.5209, 0.0498, -1.1598],
                ],
                [
                    [0.5601, -0.3231, 0.5505, -0.9595],
                    [1.3404, -0.4424, 0.8233, -2.6035],
                    [0.2673, 0.5504, 0.1273, -0.0482],
                ],
            ],
            [
                [
                    [1.6299, 1.6085, -0.0996, 0.7062],
                    [-0.3608, 1.2914, 0.8723, -0.2837],
                    [-1.2557, -0.3051, 1.0531, -0.9606],
                ],
                [
                    [-1.1698, 1.1818, -1.4536, 0.7807],
                    [0.89, 1.4763, 0.0223, -1.0139],
                    [0.519, -0.7375, -1.2078, 0.87],
                ],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.BatchNorm2d(
        num_features=2, eps=1e-05, momentum=0.1, track_running_stats=False
    ).to(device=flow.device(device))
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output, 0.0001, 0.0001))


def _test_batchnorm2d_4d_input(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [-0.8791, 0.2553, 0.7403, -0.2859],
                    [0.8006, -1.7701, -0.9617, 0.1705],
                    [0.2842, 1.7825, 0.3365, -0.8525],
                ],
                [
                    [0.7332, -0.0737, 0.7245, -0.6551],
                    [1.4461, -0.1827, 0.9737, -2.1571],
                    [0.4657, 0.7244, 0.3378, 0.1775],
                ],
            ],
            [
                [
                    [1.8896, 1.8686, 0.1896, 0.9817],
                    [-0.0671, 1.5569, 1.1449, 0.0086],
                    [-0.9468, -0.0124, 1.3227, -0.6567],
                ],
                [
                    [-0.8472, 1.3012, -1.1065, 0.9348],
                    [1.0346, 1.5703, 0.2419, -0.7048],
                    [0.6957, -0.4523, -0.8819, 1.0164],
                ],
            ],
        ],
        dtype=np.float32,
    )
    output = np.array(
        [
            [
                [
                    [-1.1868, -0.0328, 0.4606, -0.5833],
                    [0.522, -2.0933, -1.2709, -0.119],
                    [-0.0034, 1.5209, 0.0498, -1.1598],
                ],
                [
                    [0.5601, -0.3231, 0.5505, -0.9595],
                    [1.3404, -0.4424, 0.8233, -2.6035],
                    [0.2673, 0.5504, 0.1273, -0.0482],
                ],
            ],
            [
                [
                    [1.6299, 1.6085, -0.0996, 0.7062],
                    [-0.3608, 1.2914, 0.8723, -0.2837],
                    [-1.2557, -0.3051, 1.0531, -0.9606],
                ],
                [
                    [-1.1698, 1.1818, -1.4536, 0.7807],
                    [0.89, 1.4763, 0.0223, -1.0139],
                    [0.519, -0.7375, -1.2078, 0.87],
                ],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.BatchNorm2d(num_features=2, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output, 0.0001, 0.0001))


def test_batchnorm2d_infer(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [-0.8791, 0.2553, 0.7403, -0.2859],
                    [0.8006, -1.7701, -0.9617, 0.1705],
                    [0.2842, 1.7825, 0.3365, -0.8525],
                ],
                [
                    [0.7332, -0.0737, 0.7245, -0.6551],
                    [1.4461, -0.1827, 0.9737, -2.1571],
                    [0.4657, 0.7244, 0.3378, 0.1775],
                ],
            ],
            [
                [
                    [1.8896, 1.8686, 0.1896, 0.9817],
                    [-0.0671, 1.5569, 1.1449, 0.0086],
                    [-0.9468, -0.0124, 1.3227, -0.6567],
                ],
                [
                    [-0.8472, 1.3012, -1.1065, 0.9348],
                    [1.0346, 1.5703, 0.2419, -0.7048],
                    [0.6957, -0.4523, -0.8819, 1.0164],
                ],
            ],
        ],
        dtype=np.float32,
    )
    output_arr = np.array(
        [
            [
                [
                    [-0.8790956, 0.2552987, 0.7402963, -0.28589857],
                    [0.800596, -1.7700912, -0.9616952, 0.17049915],
                    [0.28419858, 1.7824911, 0.3364983, -0.85249573],
                ],
                [
                    [0.7331963, -0.07369963, 0.72449636, -0.6550967],
                    [1.4460927, -0.18269908, 0.9736951, -2.1570892],
                    [0.46569768, 0.72439635, 0.3377983, 0.1774991],
                ],
            ],
            [
                [
                    [1.8895906, 1.8685907, 0.18959905, 0.9816951],
                    [-0.06709967, 1.5568923, 1.1448942, 0.00859996],
                    [-0.9467952, -0.01239994, 1.3226933, -0.65669674],
                ],
                [
                    [-0.84719574, 1.3011935, -1.1064945, 0.9347953],
                    [1.0345949, 1.5702921, 0.24189879, -0.7047965],
                    [0.69569653, -0.45229775, -0.8818956, 1.0163949],
                ],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.BatchNorm2d(num_features=2, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    m.eval()
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output_arr, 0.0001, 0.0001))


def test_batchnorm2d_infer_4d_input(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [-0.8791, 0.2553, 0.7403, -0.2859],
                    [0.8006, -1.7701, -0.9617, 0.1705],
                    [0.2842, 1.7825, 0.3365, -0.8525],
                ],
                [
                    [0.7332, -0.0737, 0.7245, -0.6551],
                    [1.4461, -0.1827, 0.9737, -2.1571],
                    [0.4657, 0.7244, 0.3378, 0.1775],
                ],
            ],
            [
                [
                    [1.8896, 1.8686, 0.1896, 0.9817],
                    [-0.0671, 1.5569, 1.1449, 0.0086],
                    [-0.9468, -0.0124, 1.3227, -0.6567],
                ],
                [
                    [-0.8472, 1.3012, -1.1065, 0.9348],
                    [1.0346, 1.5703, 0.2419, -0.7048],
                    [0.6957, -0.4523, -0.8819, 1.0164],
                ],
            ],
        ],
        dtype=np.float32,
    )
    output_arr = np.array(
        [
            [
                [
                    [-0.8790956, 0.2552987, 0.7402963, -0.28589857],
                    [0.800596, -1.7700912, -0.9616952, 0.17049915],
                    [0.28419858, 1.7824911, 0.3364983, -0.85249573],
                ],
                [
                    [0.7331963, -0.07369963, 0.72449636, -0.6550967],
                    [1.4460927, -0.18269908, 0.9736951, -2.1570892],
                    [0.46569768, 0.72439635, 0.3377983, 0.1774991],
                ],
            ],
            [
                [
                    [1.8895906, 1.8685907, 0.18959905, 0.9816951],
                    [-0.06709967, 1.5568923, 1.1448942, 0.00859996],
                    [-0.9467952, -0.01239994, 1.3226933, -0.65669674],
                ],
                [
                    [-0.84719574, 1.3011935, -1.1064945, 0.9347953],
                    [1.0345949, 1.5702921, 0.24189879, -0.7047965],
                    [0.69569653, -0.45229775, -0.8818956, 1.0163949],
                ],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.BatchNorm2d(num_features=2, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    m.eval()
    x = flow.Tensor(input_arr, device=flow.device(device))
    y = m(x)
    test_case.assertTrue(np.allclose(y.numpy(), output_arr, 0.0001, 0.0001))


def _test_batchnorm2d_backward(test_case, device):
    input_arr = np.array(
        [
            [
                [
                    [-0.8791, 0.2553, 0.7403, -0.2859],
                    [0.8006, -1.7701, -0.9617, 0.1705],
                    [0.2842, 1.7825, 0.3365, -0.8525],
                ],
                [
                    [0.7332, -0.0737, 0.7245, -0.6551],
                    [1.4461, -0.1827, 0.9737, -2.1571],
                    [0.4657, 0.7244, 0.3378, 0.1775],
                ],
            ],
            [
                [
                    [1.8896, 1.8686, 0.1896, 0.9817],
                    [-0.0671, 1.5569, 1.1449, 0.0086],
                    [-0.9468, -0.0124, 1.3227, -0.6567],
                ],
                [
                    [-0.8472, 1.3012, -1.1065, 0.9348],
                    [1.0346, 1.5703, 0.2419, -0.7048],
                    [0.6957, -0.4523, -0.8819, 1.0164],
                ],
            ],
        ],
        dtype=np.float32,
    )
    m = flow.nn.BatchNorm2d(num_features=2, eps=1e-05, momentum=0.1).to(
        device=flow.device(device)
    )
    x = flow.Tensor(input_arr, device=flow.device(device), requires_grad=True)
    y = m(x)
    z = y.sum()
    z.backward()
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), np.zeros(shape=input_arr.shape), 1e-05, 1e-05)
    )


@flow.unittest.skip_unless_1n1d()
class TestBatchNorm(flow.unittest.TestCase):
    def test_batchnorm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_batchnorm2d,
            _test_batchnorm1d_2d_input,
            _test_batchnorm1d_3d_input,
            _test_batchnorm2d_4d_input,
            _test_batchnorm2d_track_running_stats,
            test_batchnorm2d_infer,
            test_batchnorm2d_infer_4d_input,
            _test_batchnorm2d_backward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=20, auto_backward=True)
    def test_batchnorm3d_module_with_random_data(test_case):
        channel = random().to(int)
        m = torch.nn.BatchNorm3d(num_features=channel, track_running_stats=False)
        m.train(random())
        device = "cpu"
        m.to(device)
        x = random_pytorch_tensor(ndim=5, dim1=channel, requires_grad=True).to(device)
        y = m(x)
        return y

    @unittest.skip("batchnorm module has a bug")
    def test_with_random_data(test_case):
        for device in ["cpu", "cuda"]:
            for training in [True, False]:
                test_module_against_pytorch(
                    test_case,
                    "nn.BatchNorm2d",
                    extra_annotations={
                        "num_features": int,
                        "eps": float,
                        "momentum": float,
                        "affine": bool,
                        "track_running_stats": bool,
                        "dtype": str,
                        "device": flow.device,
                    },
                    extra_generators={
                        "input": random_tensor(ndim=4, dim1=8),
                        "num_features": constant(8),
                        "eps": random(1e-06, 1),
                        "momentum": random(0, 1),
                        "track_running_stats": constant(True),
                    },
                    device=device,
                    training=training,
                    n=10,
                )

    @autotest(n=1, auto_backward=False)
    def test_batchnorm3d_module_with_random_data(test_case):
        channel = random().to(int)
        m = torch.nn.BatchNorm2d(num_features=channel, track_running_stats=False)
        m.train(random())
        device = random_device()
        m.to(device)
        x = random_pytorch_tensor(ndim=4, dim1=channel, requires_grad=False).to(device)
        y = m(x)
        return y


if __name__ == "__main__":
    unittest.main()

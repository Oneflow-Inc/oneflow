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
import oneflow.experimental.nn as nn
from test_util import GenArgList


def _test_conv3d_bias_false(test_case, device):
    np_arr = np.array(
        [[[[[ 0.30727121, -0.66000855],
           [ 0.87808079,  0.95670187]],

          [[ 0.66614127,  0.48802242],
           [-0.13199799,  0.43449000]]]]]
    )
    input = flow.Tensor(
        np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
    )
    weight = np.array(
        [[[[[ 1.87088445e-01,  3.68028134e-02, -1.44043714e-01],
           [ 1.52515009e-01, -1.72857031e-01, -2.36443579e-02],
           [ 1.24335140e-02,  1.30840793e-01, -1.73997611e-01]],

          [[-9.25847888e-03,  5.28722703e-02,  9.00785476e-02],
           [ 1.26786396e-01, -4.75969464e-02, -5.22580743e-02],
           [ 4.09790725e-02,  1.60206333e-01, -1.62584454e-01]],

          [[-4.69704717e-02, -9.83292907e-02, -9.13341194e-02],
           [ 1.66149214e-01,  5.57178110e-02, -1.27344340e-01],
           [ 1.13941595e-01,  2.73731053e-02, -4.22969013e-02]]]],

        [[[[ 9.55034941e-02,  8.57733339e-02, -8.60577524e-02],
           [-6.73041493e-02,  4.27209437e-02,  3.94689292e-02],
           [-8.51862356e-02,  1.00283489e-01, -1.15714416e-01]],

          [[ 9.75115597e-03, -2.14173943e-02, -1.00418925e-04],
           [ 1.88246951e-01,  1.77234724e-01, -4.05247211e-02],
           [-9.39041376e-02, -6.59874976e-02, -1.24241456e-01]],

          [[ 1.62340745e-01, -8.17814395e-02, -1.59555525e-01],
           [-3.44597697e-02,  1.02577761e-01,  8.94055516e-02],
           [-9.51993614e-02, -4.49914485e-02, -5.05818725e-02]]]],

        [[[[-1.70308709e-01, -1.64391235e-01,  1.04269519e-01],
           [ 8.42359513e-02, -1.16145238e-01, -1.00101531e-01],
           [ 3.33324075e-02, -1.35232419e-01, -5.58677763e-02]],

          [[ 1.02465704e-01, -9.81703401e-04,  6.56596571e-02],
           [ 7.83573538e-02, -1.71904668e-01,  9.85644013e-02],
           [ 4.36134040e-02, -1.40337408e-01, -1.33946016e-01]],

          [[ 1.72919169e-01, -1.03741437e-01, -1.57389179e-01],
           [-1.07288539e-01,  1.00711629e-01, -1.20750420e-01],
           [-1.39835849e-01, -1.74738079e-01,  1.67629346e-01]]]]]
    )
    m = nn.Conv3d(1, 3, 3, padding=1, stride=1, bias=False)
    m.weight = flow.nn.Parameter(flow.Tensor(weight))
    m = m.to(device)
    output = m(input)
    np_out = np.array(
        [[[[[-0.02027042,  0.37084281,  0.08817714],
           [-0.24862868,  0.69746441, -0.17246124],
           [ 0.03038912,  0.16950776,  0.08093131]],

          [[-0.14037848,  0.48571134,  0.03124815],
           [-0.14777701,  0.49216452, -0.27985838],
           [ 0.22745541,  0.20416659, -0.02639692]],

          [[-0.31470472,  0.47001129,  0.00401691],
           [ 0.45371252,  0.30338943, -0.15259278],
           [-0.03378177,  0.88987046,  0.03981188]]],


         [[[-0.03364231, -0.16610493, -0.11074042],
           [ 0.06231878,  0.48245382,  0.19843927],
           [-0.26843220,  0.19641112, -0.01014443]],

          [[-0.01202443,  0.03920723, -0.12517187],
           [ 0.16568711,  0.94875181,  0.03210684],
           [-0.05752502, -0.53306884, -0.33480555]],

          [[ 0.03931380,  0.00930462, -0.18695907],
           [ 0.28651753,  0.34585327,  0.39660025],
           [ 0.23038211,  0.13142151, -0.61334491]]],


         [[[-0.13145924, -0.48609203, -0.00221704],
           [-0.32076758,  0.32790440,  0.30248162],
           [ 0.10819200, -0.10569721,  0.62180424]],

          [[-0.19936201, -0.13873456, -0.01971424],
           [-0.77115691, -0.21294948,  0.93342757],
           [ 0.00689149, -0.13903564,  0.23208751]],

          [[-0.41609871, -0.00349248,  0.01258709],
           [-0.15659288,  0.61498058,  0.46487308],
           [-0.27688783, -0.00568488,  0.37577158]]]]]
    )
    test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
    output = output.sum()
    output.backward()
    np_grad = np.array(
        [[[[[ 0.47818899,  0.42949945, -0.34857965],
           [ 0.47923028, -0.33581090, -1.06515801],
           [ 0.27533060, -0.56951642, -1.08362198]],

          [[ 0.76603436,  0.15037680, -0.94039267],
           [ 0.45362562, -0.85363305, -1.77457690],
           [ 0.24528866, -0.68349695, -1.10090983]],

          [[ 0.77240074,  0.36685210, -0.44218731],
           [ 0.40352046, -0.34804949, -1.02668369],
           [ 0.26565164, -0.23327729, -0.52066362]]]]]
    )
    test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


# def _test_conv1d_bias_true(test_case, device):
#     np_arr = np.array(
        
#     )
#     input = flow.Tensor(
#         np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
#     )
#     weight = np.array(
#     )
#     bias = np.array()
#     m = nn.Conv1d(2, 4, 3, stride=1, bias=True)
#     m.weight = flow.nn.Parameter(flow.Tensor(weight))
#     m.bias = flow.nn.Parameter(flow.Tensor(bias))
#     m = m.to(device)
#     np_out = np.array(
        
#     )
#     output = m(input)
#     test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
#     output = output.sum()
#     output.backward()
#     np_grad = np.array(
        
#     )
#     test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


# def _test_conv1d_dilation(test_case, device):
#     np_arr = np.array(
#     )
#     input = flow.Tensor(
#         np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
#     )
#     weight = np.array(
        
#     )
#     m = nn.Conv1d(1, 3, 3, stride=1, bias=False)
#     m.weight = flow.nn.Parameter(flow.Tensor(weight))
#     m = m.to(device)
#     output = m(input)
#     np_out = np.array(
        
#     )
#     test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
#     output = output.sum()
#     output.backward()
#     np_grad = np.array(
#     )
#     test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


# def _test_conv1d_stride(test_case, device):
#     np_arr = np.array(
#     )
#     input = flow.Tensor(
#         np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
#     )
#     weight = np.array(
        
#     )
#     m = nn.Conv1d(1, 3, 3, stride=2, bias=False)
#     m.weight = flow.nn.Parameter(flow.Tensor(weight))
#     m = m.to(device)
#     output = m(input)
#     np_out = np.array(
        
#     )
#     test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
#     output = output.sum()
#     output.backward()
#     np_grad = np.array(
#     )
#     test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


# def _test_conv1d_group_bias_true(test_case, device):
#     np_arr = np.array(
        
#     )
#     input = flow.Tensor(
#         np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
#     )
#     weight = np.array(
        
#     )
#     bias = np.array()
#     m = nn.Conv1d(2, 4, 3, groups=2, stride=1, bias=True)
#     m.weight = flow.nn.Parameter(flow.Tensor(weight))
#     m.bias = flow.nn.Parameter(flow.Tensor(bias))
#     m = m.to(device)
#     np_out = np.array(
#     )
#     output = m(input)
#     test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
#     output = output.sum()
#     output.backward()
#     np_grad = np.array(
        
#     )
#     test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


# def _test_conv1d_group_large_out_bias_true(test_case, device):
#     np_arr = np.array(
        
#     )
#     input = flow.Tensor(
#         np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
#     )
#     weight = np.array(
        
#     )
#     bias = np.array(
#     )
#     m = nn.Conv1d(2, 6, 3, groups=2, stride=1, bias=True)
#     m.weight = flow.nn.Parameter(flow.Tensor(weight))
#     m.bias = flow.nn.Parameter(flow.Tensor(bias))
#     m = m.to(device)
#     np_out = np.array(
        
#     )
#     output = m(input)
#     test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
#     output = output.sum()
#     output.backward()
#     np_grad = np.array(
        
#     )
#     test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


# def _test_conv1d_group_large_in_bias_true(test_case, device):
#     np_arr = np.array(
        
#     )
#     input = flow.Tensor(
#         np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
#     )
#     weight = np.array(
        
#     )
#     bias = np.array()
#     m = nn.Conv1d(4, 2, 3, groups=2, stride=1, bias=True)
#     m.weight = flow.nn.Parameter(flow.Tensor(weight))
#     m.bias = flow.nn.Parameter(flow.Tensor(bias))
#     m = m.to(device)
#     np_out = np.array(
#     )
#     output = m(input)
#     test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
#     output = output.sum()
#     output.backward()
#     np_grad = np.array(
        
#     )
#     test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


# def _test_conv1d_compilcate(test_case, device):
#     np_arr = np.array(
#     )
#     input = flow.Tensor(
#         np_arr, dtype=flow.float32, device=flow.device(device), requires_grad=True
#     )
#     weight = np.array(
        
#     )
#     bias = np.array()
#     m = nn.Conv1d(4, 4, 3, groups=2, stride=2, padding=2, dilation=2, bias=True)
#     m.weight = flow.nn.Parameter(flow.Tensor(weight))
#     m.bias = flow.nn.Parameter(flow.Tensor(bias))
#     m = m.to(device)
#     np_out = np.array(
        
#     )
#     output = m(input)
#     test_case.assertTrue(np.allclose(output.numpy(), np_out, 1e-6, 1e-6))
#     output = output.sum()
#     output.backward()
#     np_grad = np.array(
        
#     )
#     test_case.assertTrue(np.allclose(input.grad.numpy(), np_grad, 1e-6, 1e-6))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestConv3d(flow.unittest.TestCase):
    def test_conv3d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            # _test_conv1d_bias_true,
            _test_conv3d_bias_false,
            # _test_conv1d_dilation,
            # _test_conv1d_stride,
            # _test_conv1d_group_bias_true,
            # _test_conv1d_group_large_out_bias_true,
            # _test_conv1d_group_large_in_bias_true,
            # _test_conv1d_compilcate,
        ]
        arg_dict["device"] = ["cuda", "cpu"]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
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
import math

import oneflow.experimental as flow
from test_util import GenArgList


def _nd_tuple_to_dhw(nd_tuple, dim, prefix=1, dhw_offset=0):
    assert dim <= 3
    assert dim == len(nd_tuple) - dhw_offset
    nd_tuple = list(nd_tuple)
    dhw_tuple = nd_tuple[:dhw_offset]
    dhw_tuple.extend([prefix for _ in range(3 - dim)])
    dhw_tuple.extend(nd_tuple[dhw_offset:])
    return tuple(dhw_tuple)


def _dhw_tuple_to_nd(dhw_tuple, dim, prefix=1, dhw_offset=0):
    assert dim <= 3
    assert 3 == len(dhw_tuple) - dhw_offset
    dhw_tuple = list(dhw_tuple)
    nd_tuple = dhw_tuple[:dhw_offset]
    nd_offset = dhw_offset + 3 - dim
    for i in dhw_tuple[dhw_offset:nd_offset]:
        assert prefix == i
    nd_tuple.extend(dhw_tuple[nd_offset:])
    return tuple(nd_tuple)


class MaxPoolNumpy:
    def __init__(self, dim=2, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)):
        self.dim = dim
        self.stride = _nd_tuple_to_dhw(stride, dim)
        self.padding = _nd_tuple_to_dhw(padding, dim, prefix=0)
        self.kernel_size = _nd_tuple_to_dhw(kernel_size, dim)
        self.w_depth = self.kernel_size[0]
        self.w_height = self.kernel_size[1]
        self.w_width = self.kernel_size[2]
        self.min_val = np.finfo(np.float64).min

    def __call__(self, x):
        self.x_shape = x.shape
        x_shape_5d = _nd_tuple_to_dhw(self.x_shape, self.dim, prefix=1, dhw_offset=2)
        x = x.reshape(x_shape_5d)
        self.in_batch = np.shape(x)[0]
        self.in_channel = np.shape(x)[1]
        self.in_depth = np.shape(x)[2]
        self.in_height = np.shape(x)[3]
        self.in_width = np.shape(x)[4]

        pad_x = np.pad(
            x,
            (
                (0, 0),
                (0, 0),
                (self.padding[0], self.padding[0]),
                (self.padding[1], self.padding[1]),
                (self.padding[2], self.padding[2]),
            ),
            "constant",
            constant_values=(self.min_val, self.min_val),
        )
        self.pad_x = pad_x
        self.pad_shape = pad_x.shape

        self.out_depth = int((self.in_depth - self.w_depth) / self.stride[0]) + 1
        self.out_height = int((self.in_height - self.w_height) / self.stride[1]) + 1
        self.out_width = int((self.in_width - self.w_width) / self.stride[2]) + 1
        self.pad_out_depth = np.uint16(
            math.ceil((self.pad_shape[2] - self.w_depth + 1) / self.stride[0])
        )
        self.pad_out_height = np.uint16(
            math.ceil((self.pad_shape[3] - self.w_height + 1) / self.stride[1])
        )
        self.pad_out_width = np.uint16(
            math.ceil((self.pad_shape[4] - self.w_width + 1) / self.stride[2])
        )

        out = np.zeros(
            (
                self.in_batch,
                self.in_channel,
                self.pad_out_depth,
                self.pad_out_height,
                self.pad_out_width,
            )
        )
        self.arg_max = np.zeros_like(out, dtype=np.int32)
        for n in range(self.in_batch):
            for c in range(self.in_channel):
                for i in range(self.pad_out_depth):
                    for j in range(self.pad_out_height):
                        for k in range(self.pad_out_width):
                            start_i = i * self.stride[0]
                            start_j = j * self.stride[1]
                            start_k = k * self.stride[2]
                            end_i = start_i + self.w_depth
                            end_j = start_j + self.w_height
                            end_k = start_k + self.w_width
                            out[n, c, i, j, k] = np.max(
                                pad_x[n, c, start_i:end_i, start_j:end_j, start_k:end_k]
                            )
                            self.arg_max[n, c, i, j, k] = np.argmax(
                                pad_x[n, c, start_i:end_i, start_j:end_j, start_k:end_k]
                            )

        self.out_shape_5d = out.shape
        out_shape = _dhw_tuple_to_nd(out.shape, self.dim, dhw_offset=2)
        out = out.reshape(out_shape)
        return out

    def backward(self, d_loss):
        d_loss = d_loss.reshape(self.out_shape_5d)
        dx = np.zeros_like(self.pad_x)
        for n in range(self.in_batch):
            for c in range(self.in_channel):
                for i in range(self.pad_out_depth):
                    for j in range(self.pad_out_height):
                        for k in range(self.pad_out_width):
                            start_i = i * self.stride[0]
                            start_j = j * self.stride[1]
                            start_k = k * self.stride[2]
                            end_i = start_i + self.w_depth
                            end_j = start_j + self.w_height
                            end_k = start_k + self.w_width
                            index = np.unravel_index(
                                self.arg_max[n, c, i, j, k], self.kernel_size
                            )
                            dx[n, c, start_i:end_i, start_j:end_j, start_k:end_k][
                                index
                            ] += d_loss[n, c, i, j, k]
        dx = dx[
            :,
            :,
            self.padding[0] : self.pad_shape[2] - self.padding[0],
            self.padding[1] : self.pad_shape[3] - self.padding[1],
            self.padding[2] : self.pad_shape[4] - self.padding[2],
        ]
        dx = dx.reshape(self.x_shape)
        return dx


def _test_maxpool2d(test_case, device):
    dim = 2
    
    input_arr = np.random.randn(2, 3, 4, 5)
    kernel_size, stride, padding = (3, 3), (1, 1), (1, 1)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, device=flow.device(device))
    output, indice = m(x)
    test_case.assertTrue(indice.shape == x.shape)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))


def _test_maxpool2d_ceil_mode(test_case, device):
    dim = 2
    input_arr = np.array(
        [[[[-0.89042996,  2.33971243, -0.86660827,  0.80398747],
         [-1.46769364, -0.78125064,  1.50086563, -0.76278226],
         [ 1.31984534,  0.20741192, -0.86507054, -0.40776015],
         [-0.89910823,  0.44932938,  1.49148118, -0.22036761]],

        [[-0.5452334 , -0.10255169, -1.42035108,  0.73922913],
         [-0.03192764,  0.69341935,  0.96263152, -1.52070843],
         [ 0.02058239,  1.504032  ,  1.84423001, -0.0130596 ],
         [ 2.20517719,  0.38449598,  0.85677771,  0.60425179]],

        [[-1.64366213,  0.51370298, -0.21754866, -0.05085382],
         [ 1.17065374,  1.13857674, -1.13070507,  0.44353707],
         [-1.30783846, -0.48031445,  0.41807536, -2.13778887],
         [ 0.08259005,  0.5798125 ,  0.03024696,  1.96100924]]],


       [[[ 0.45173843, -0.34680027, -0.99754943,  0.18539502],
         [-0.68451047, -0.03217399,  0.44705642, -0.39016231],
         [-0.18062337,  1.82099303, -0.19113869,  0.85298683],
         [ 0.14080452,  0.15306701, -1.02466827, -0.34480665]],

        [[-0.21048489,  0.20933038, -0.09206508, -1.80402519],
         [-0.52028985,  0.01140166, -1.13452858,  0.96648332],
         [ 0.26454393,  0.48343972, -1.84055509, -0.01256443],
         [ 0.31024029,  0.11983007,  0.98806488,  0.93557438]],

        [[ 0.39152445,  0.672159  ,  0.71289289, -0.68072016],
         [ 0.33711062, -1.78106242,  0.34545201, -1.62029359],
         [ 0.47343899, -2.3433269 , -0.44517497,  0.09004267],
         [ 0.26310742, -1.53121271,  0.65028836,  1.3669488 ]]]]
    )

    ceil_mode_out = np.array(
        [[[[ 2.33971243,  2.33971243,  0.80398747],
         [ 1.31984534,  1.50086563, -0.22036761],
         [ 0.44932938,  1.49148118, -0.22036761]],

        [[ 0.69341935,  0.96263152,  0.73922913],
         [ 2.20517719,  1.84423001,  0.60425179],
         [ 2.20517719,  0.85677771,  0.60425179]],

        [[ 1.17065374,  1.13857674,  0.44353707],
         [ 1.17065374,  1.96100924,  1.96100924],
         [ 0.5798125 ,  1.96100924,  1.96100924]]],


       [[[ 0.45173843,  0.44705642,  0.18539502],
         [ 1.82099303,  1.82099303,  0.85298683],
         [ 0.15306701,  0.15306701, -0.34480665]],

        [[ 0.20933038,  0.96648332,  0.96648332],
         [ 0.48343972,  0.98806488,  0.96648332],
         [ 0.31024029,  0.98806488,  0.93557438]],

        [[ 0.672159  ,  0.71289289, -0.68072016],
         [ 0.47343899,  1.3669488 ,  1.3669488 ],
         [ 0.26310742,  1.3669488 ,  1.3669488 ]]]]
    )
    kernel_size, stride, padding = (3, 3), (2, 2), (1, 1)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m1 = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=False)
    m2 = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=True)
    m1.to(flow.device(device))
    m2.to(flow.device(device))
    x = flow.Tensor(input_arr, device=flow.device(device))
    output1 = m1(x)
    output2 = m2(x)
    test_case.assertTrue(np.allclose(numpy_output, output1.numpy(), 1e-4, 1e-4))
    test_case.assertTrue(np.allclose(ceil_mode_out, output2.numpy(), 1e-4, 1e-4))


def _test_maxpool2d_special_kernel_size(test_case, device):
    dim = 2
    input_arr = np.random.randn(1, 1, 6, 6)
    kernel_size, stride, padding = (1, 1), (5, 5), (0, 0)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, device=flow.device(device))
    output = m(x)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))


def _test_maxpool2d_diff_kernel_stride(test_case, device):
    dim = 2
    input_arr = np.random.randn(9, 7, 32, 20)
    kernel_size, stride, padding = (2, 4), (4, 5), (1, 2)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, device=flow.device(device))
    output = m(x)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))


def _test_maxpool2d_negative_input(test_case, device):
    dim = 2
    input_arr = -1.23456 * np.ones((1, 1, 1, 1), dtype=np.float)
    kernel_size, stride, padding = (5, 5), (5, 5), (2, 2)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, device=flow.device(device))
    output = m(x)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))


def _test_maxpool2d_backward(test_case, device):
    dim = 2
    input_arr = np.random.randn(6, 4, 7, 9)
    kernel_size, stride, padding = (4, 4), (1, 1), (1, 2)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)

    output = output.sum()
    output.backward()
    doutput = np.ones_like(numpy_output, dtype=np.float64)
    numpy_grad = m_numpy.backward(doutput)
    test_case.assertTrue(np.allclose(x.grad.numpy(), numpy_grad, 1e-1, 1e-1))


def _test_maxpool2d_special_kernel_size_backward(test_case, device):
    dim = 2
    input_arr = np.random.randn(1, 1, 6, 6)
    kernel_size, stride, padding = (1, 1), (5, 5), (0, 0)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)

    output = output.sum()
    output.backward()
    doutput = np.ones_like(numpy_output, dtype=np.float64)
    numpy_grad = m_numpy.backward(doutput)
    test_case.assertTrue(np.allclose(x.grad.numpy(), numpy_grad, 1e-5, 1e-5))


def _test_maxpool2d_diff_kernel_stride_backward(test_case, device):
    dim = 2
    input_arr = np.random.randn(9, 7, 32, 20)
    kernel_size, stride, padding = (2, 4), (4, 5), (1, 2)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)

    output = output.sum()
    output.backward()
    doutput = np.ones_like(numpy_output, dtype=np.float64)
    numpy_grad = m_numpy.backward(doutput)
    test_case.assertTrue(np.allclose(x.grad.numpy(), numpy_grad, 1e-5, 1e-5))


def _test_maxpool2d_negative_input_backward(test_case, device):
    dim = 2
    input_arr = -1.23456 * np.ones((1, 1, 1, 1), dtype=np.float)
    kernel_size, stride, padding = (5, 5), (5, 5), (2, 2)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)

    output = output.sum()
    output.backward()
    doutput = np.ones_like(numpy_output, dtype=np.float64)
    numpy_grad = m_numpy.backward(doutput)
    test_case.assertTrue(np.allclose(x.grad.numpy(), numpy_grad, 1e-5, 1e-5))


def _test_maxpool3d_backward(test_case, device):
    dim = 3
    input_arr = np.random.randn(6, 4, 8, 7, 9)
    kernel_size, stride, padding = (4, 4, 4), (1, 1, 1), (2, 1, 2)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))

    output = output.sum()
    output.backward()
    doutput = np.ones_like(numpy_output, dtype=np.float64)
    numpy_grad = m_numpy.backward(doutput)
    test_case.assertTrue(np.allclose(x.grad.numpy(), numpy_grad, 1e-5, 1e-5))


def _test_maxpool3d_special_kernel_size_backward(test_case, device):
    dim = 3
    input_arr = np.random.randn(1, 1, 6, 6, 6)
    kernel_size, stride, padding = (1, 1, 1), (5, 5, 5), (0, 0, 0)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))

    output = output.sum()
    output.backward()
    doutput = np.ones_like(numpy_output, dtype=np.float64)
    numpy_grad = m_numpy.backward(doutput)
    test_case.assertTrue(np.allclose(x.grad.numpy(), numpy_grad, 1e-5, 1e-5))


def _test_maxpool3d_diff_kernel_stride_backward(test_case, device):
    dim = 3
    input_arr = np.random.randn(9, 7, 48, 32, 20)
    kernel_size, stride, padding = (6, 2, 3), (5, 4, 5), (4, 1, 2)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))

    output = output.sum()
    output.backward()
    doutput = np.ones_like(numpy_output, dtype=np.float64)
    numpy_grad = m_numpy.backward(doutput)
    test_case.assertTrue(np.allclose(x.grad.numpy(), numpy_grad, 1e-5, 1e-5))


def _test_maxpool3d_negative_input_backward(test_case, device):
    dim = 3
    input_arr = -1.23456 * np.ones((1, 1, 1, 1, 1), dtype=np.float)
    kernel_size, stride, padding = (5, 5, 5), (5, 5, 5), (2, 2, 2)

    m_numpy = MaxPoolNumpy(dim, kernel_size, stride, padding)
    numpy_output = m_numpy(input_arr)

    m = flow.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
    m.to(flow.device(device))
    x = flow.Tensor(input_arr, requires_grad=True, device=flow.device(device))
    output = m(x)
    test_case.assertTrue(np.allclose(numpy_output, output.numpy(), 1e-4, 1e-4))

    output = output.sum()
    output.backward()
    doutput = np.ones_like(numpy_output, dtype=np.float64)
    numpy_grad = m_numpy.backward(doutput)
    test_case.assertTrue(np.allclose(x.grad.numpy(), numpy_grad, 1e-5, 1e-5))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestPoolingModule(flow.unittest.TestCase):
    def test_maxpool2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_maxpool2d,
            _test_maxpool2d_ceil_mode,
            _test_maxpool2d_special_kernel_size,
            _test_maxpool2d_diff_kernel_stride,
            _test_maxpool2d_negative_input,
            _test_maxpool2d_backward,
            _test_maxpool2d_special_kernel_size_backward,
            _test_maxpool2d_diff_kernel_stride_backward,
            _test_maxpool2d_negative_input_backward,
        ]
        
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    # def test_maxpool3d(test_case):
    #     arg_dict = OrderedDict()
    #     arg_dict["test_fun"] = [
    #         _test_maxpool3d_backward,
    #         _test_maxpool3d_special_kernel_size_backward,
    #         _test_maxpool3d_diff_kernel_stride_backward,
    #         _test_maxpool3d_negative_input_backward,
    #     ]
    #     arg_dict["device"] = ["cpu", "cuda"]
    #     for arg in GenArgList(arg_dict):
    #         arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

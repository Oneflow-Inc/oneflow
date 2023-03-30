import os

import numpy as np
import torch

import oneflow as flow


def test_summation_real():
    input_shape = (3,5,2)
    dtype = np.float32
    x = np.random.randn(*input_shape)
    x = x.astype(dtype)

    y = np.random.randn(*input_shape)
    y = y.astype(dtype)

    x_flow = flow.from_numpy(x).requires_grad_(True)
    y_flow = flow.from_numpy(y).requires_grad_(True)

    # ret = x_flow.sum()
    # ret = ret.requires_grad_(True)
    # ret.backward()
    # exit(0)

    ret = x_flow + y_flow
    ret = ret.sum()
    ret.backward()

    exit(0)

def test_summation_complex():
    input_shape = (3,5,2)
    # input_shape = (1,)
    dtype = np.complex64
    x = np.random.randn(*input_shape) + 1.j * np.random.randn(*input_shape)
    x = x.astype(dtype)

    y = np.random.randn(*input_shape) + 1.j * np.random.randn(*input_shape)
    y = y.astype(dtype)

    x_flow = flow.from_numpy(x).requires_grad_(True)
    y_flow = flow.from_numpy(y).requires_grad_(True)

    x_torch = torch.from_numpy(x).requires_grad_(True)
    y_torch = torch.from_numpy(y).requires_grad_(True)

    ret_torch = x_torch * y_torch
    ret_torch = ret_torch.sum()
    ret_torch.backward()
    
    # x_torch_grad = x_torch.grad.detach().cpu()
    # y_torch = y_torch.detach().cpu()
    # ret = x_flow.sum()
    # ret = ret.requires_grad_(True)
    # ret.backward()
    # exit(0)

    ret = x_flow * y_flow
    ret = ret.sum()
    ret.backward()

    # x_flow_grad = x_flow.grad.detach().cpu()
    # y_flow = y_flow.detach().cpu()

    exit(0)
    # requires grad
    x = flow.randn(3,5,3).requires_grad_(True)
    y = flow.randn(3,5,3).requires_grad_(True)
    ret = x + y
    ret = ret.sum()
    ret.backward()
    print("stop here")

def test_fft():
    
    # t4d = flow.empty(3, 3, 4, 2)
    # p1d = (1, 1)
    # out = flow._C.pad(t4d, p1d)
    
    # np_dtype = np.complex64
    # c = [
    #     [3.14 + 2j, 3.14 + 2j],
    #     [3.14 + 2j, 3.14 + 2j],
    #     [3.14 + 2j, 3.14 + 2j],
    # ]
    # np_c = np.random.randn(5,2,3, dtype=np_dtype)
    # np_c = np.array(c, dtype=np_dtype)
    
    shape = (3,5,4)
    c_torch = torch.randn(shape, dtype=torch.complex64)
    ret_torch = torch.fft.fft(c_torch, dim=0).numpy()
    print(ret_torch)

    np_c = c_torch.numpy()
    c_flow = flow.from_numpy(np_c)
    ret_flow = flow._C.fft(c_flow, dim=0).numpy()
    print(ret_flow)
    diff = np.linalg.norm(ret_torch - ret_flow).sum()
    print("diff = ", diff)

    # c = flow.from_numpy(np_c)
    # ret = flow._C.fft(c, dim=0)

if __name__ == "__main__":
    # test_fft()
    test_summation_complex()
    # test_summation_real()
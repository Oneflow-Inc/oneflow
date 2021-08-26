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
from test_util import GenArgList

from automated_test_util import *
import oneflow as flow


def np_distance(x1, x2, eps, p):
    d_norm = np.power(np.abs(x1 - x2 + eps), p)
    d = np.power(np.sum(d_norm, axis=-1), 1.0 / p)
    return d


def np_triplet_margin_loss(np_anchor, np_pos, np_neg, eps, np_p, swap, redution, np_margin):
    np_d_ap = np_distance(np_anchor, np_pos, eps, np_p)
    np_d_an = np_distance(np_anchor, np_neg, eps, np_p)
    if swap:
        np_dist_swap = np_distance(np_pos, np_neg, eps, np_p)
        np_d_an = np.minimum(np_d_an, np_dist_swap)

    np_triplet_marginloss = np.maximum(np_d_ap - np_d_an + np_margin, 0)
    if redution == "none":
        np_triplet_marginloss = np_triplet_marginloss
    elif redution == "mean":
        np_triplet_marginloss = np.mean(np_triplet_marginloss)
    elif redution == "sum":
        np_triplet_marginloss = np.sum(np_triplet_marginloss)
    return np_triplet_marginloss


def np_compute_per_diff(x1, x2, p, eps):
    _abs_index = np.where(x1 - x2 > 0, 1, -1)
    _abs_index_support = np.where(x1 - x2 == 0, 1, 0)
    _abs_grad = _abs_index + _abs_index_support
    _abs_val = np.abs(x1 - x2 + eps)
    _power_abs_val = np.power(_abs_val, p)
    _sum_val = np.sum(_power_abs_val, axis=-1, keepdims=True)
    _sqrt_sum_val = np.power(_sum_val + eps, 1.0 / p - 1)
    _power_val = np.power(_abs_val, p - 1)
    _grad = np.multiply(_sqrt_sum_val, _power_val)
    _grad *= _abs_grad
    return _grad / x1.shape[0]


def np_triplet_margin_loss_grad(np_anchor, np_pos, np_neg, np_p, reduction, eps, margin, swap):
    np_pos_grad = np_compute_per_diff(np_anchor, np_pos, np_p, eps)
    np_neg_grad = np_compute_per_diff(np_anchor, np_neg, np_p, eps)
    total_grad = np_pos_grad - np_neg_grad
    
    np_d_ap = np_distance(np_anchor, np_pos, eps, np_p)
    np_d_an = np_distance(np_anchor, np_neg, eps, np_p)
    d = np_d_ap - np_d_an + margin

    zero_index = np.where(d < -1e-06)
    for i in zero_index:
        total_grad[i] = 0
    np_d_pn = np_distance(np_pos, np_neg, eps, np_p)
    
    if swap:      
        elem_cnt = np_d_an.size
        np_d_pn = np_d_pn.reshape(-1)
        np_d_an = np_d_an.reshape(-1)
        print("old_total_grad:",total_grad)
        for i in np.arange(elem_cnt):
            if np_d_pn[i] < np_d_an[i]:
                print("np_d_an[i]:",np_d_an[i], "np_d_pn[i]:",np_d_pn[i])
                print("np_pos_grad[i]:",np_pos_grad[i])
                print("total_grad[i]:",total_grad[i])
                total_grad[i] = np_pos_grad[i]
   
    if reduction == "none":
        grad = total_grad * np_anchor.shape[0]
    elif reduction == "mean":
        grad = total_grad
    elif reduction == "sum":
        grad = total_grad * np_anchor.shape[0]
    return grad


def _test_tripletmarginloss(test_case, shape, margin, swap, p, reduction, device):
    anchor = np.random.random(size = shape).astype(np.float32)
    pos = np.random.random(size=shape).astype(np.float32)
    neg = np.random.random(size=shape).astype(np.float32)
    of_anchor = flow.Tensor(anchor, dtype=flow.float32, device=flow.device(device),requires_grad=True)
    of_pos = flow.Tensor(pos, dtype=flow.float32, device=flow.device(device), requires_grad=True)
    of_neg = flow.Tensor(neg, dtype=flow.float32, device=flow.device(device), requires_grad=True)
    eps = 1e-6
    np_triplet_marginloss = np_triplet_margin_loss(anchor, pos, neg, eps, p, swap, reduction, margin)
    triplet_marginloss = flow.nn.TripletMarginLoss(margin=margin, p=p, eps=eps, swap=swap, reduction = reduction)
    output_triplet_marginloss = triplet_marginloss(of_anchor, of_pos, of_neg)
    test_case.assertTrue(np.allclose(output_triplet_marginloss.numpy(), np_triplet_marginloss, 1e-06, 1e-06))
    print("@@@@shape:",shape, "swap:",swap,"p",p, "reduction",reduction, "device:",device)

    output_triplet_marginloss = output_triplet_marginloss.sum()
    output_triplet_marginloss.backward()
    print("flow_anchor.grad:",of_anchor.grad)
   
    np_grad = np_triplet_margin_loss_grad(anchor, pos, neg, p, reduction, eps, margin, swap)
    
    print("np_autor_grad:",np_grad)
    test_case.assertTrue(np.allclose(of_anchor.grad.numpy(), np_grad, 0.002, 0.002))
    

@flow.unittest.skip_unless_1n1d()
class TestTripletMarginLoss(flow.unittest.TestCase):
    
    def test_triplet_margin_loss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_tripletmarginloss,
        ]
        arg_dict["shape"] = [(2, 3), (3, 3), (3, 4)]
        arg_dict["margin"] = [1.0,]
        arg_dict["swap"] = [False, True]
        arg_dict["p"] = [1, 2, 3]
        arg_dict["reduction"] = ["mean", "sum","none"]
        arg_dict["device"] = ["cpu","cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

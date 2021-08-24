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


def np_triplet_margin_loss(np_anchor, np_pos, np_neg, eps, np_p, swap, redution, np_margin):
    np_d_ap_tmp = np.power(np.abs(np_anchor - np_pos + eps), np_p)
    np_d_an_tmp = np.power(np.abs(np_anchor - np_neg + eps), np_p)
    np_d_ap = np.power(np.sum(np_d_ap_tmp, axis=-1), 1.0 / np_p)
    np_d_an = np.power(np.sum(np_d_an_tmp, axis=-1), 1.0 / np_p) 
    if swap:
        np_dist_swap = np.power(np.abs(np_pos - np_neg + eps), np_p)
        np_dist_swap = np.power(np.sum(np_dist_swap, axis=-1), 1.0 / np_p)
        np_d_an = np.minimum(np_d_an, np_dist_swap)

    np_triplet_marginloss = np.maximum(np_d_ap - np_d_an + np_margin, 0)
    if redution == "none":
        np_triplet_marginloss = np_triplet_marginloss
    elif redution == "mean":
        np_triplet_marginloss = np.mean(np_triplet_marginloss)
    elif redution == "sum":
        np_triplet_marginloss = np.sum(np_triplet_marginloss)
    return np_triplet_marginloss


def _test_tripletmarginloss(test_case, shape, margin, swap, p, reduction, device):
    anchor = np.random.random(size = shape).astype(np.float32)
    pos = np.random.random(size=shape).astype(np.float32)
    neg = np.random.random(size=shape).astype(np.float32)
    of_anchor = flow.Tensor(anchor, dtype=flow.float32, device=flow.device(device))
    of_pos = flow.Tensor(pos, dtype=flow.float32, device=flow.device(device))
    of_neg = flow.Tensor(neg, dtype=flow.float32, device=flow.device(device))
    eps = 1e-6
    np_triplet_marginloss = np_triplet_margin_loss(anchor, pos, neg, eps, p, swap, reduction, margin)
    triplet_marginloss = flow.nn.TripletMarginLoss(margin=margin, p=p, eps=eps, swap=swap, reduction = reduction)
    output_triplet_marginloss = triplet_marginloss(of_anchor, of_pos, of_neg)
    test_case.assertTrue(np.allclose(output_triplet_marginloss.numpy(), np_triplet_marginloss, 1e-06, 1e-06))


@flow.unittest.skip_unless_1n1d()
class TestTripletMarginLoss(flow.unittest.TestCase):
    def test_triplet_margin_loss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_tripletmarginloss,
        ]
        arg_dict["shape"] = [(2, 3), (2, 4, 5, 6)]
        arg_dict["margin"] = [1.0,]
        arg_dict["swap"] = [False, True]
        arg_dict["p"] = [1, 2, 3]
        arg_dict["reduction"] = ["none", "mean", "sum"]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

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
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import oneflow.unittest

from oneflow.test_utils.test_util import GenArgList

import torch as torch_original
from oneflow.test_utils.automated_test_util import *


def _test_spectral_norm(test_case, device):
    input = flow.rand(3, 5).to(device)
    m = nn.Linear(5, 7).to(device)
    m = flow.nn.utils.spectral_norm(m)

    test_case.assertEqual(m.weight_u.size(), flow.Size([m.weight.size(0)]))

    test_case.assertTrue(hasattr(m, "weight_orig"))
    test_case.assertTrue("weight_orig" in m._parameters)

    test_case.assertTrue(hasattr(m, "weight_u"))
    test_case.assertTrue("weight_u" in m._buffers)
    test_case.assertTrue("weight_v" in m._buffers)

    test_case.assertFalse("weight" in m._parameters)
    test_case.assertFalse("weight" in m._buffers)

    test_case.assertEqual(m.weight_orig.storage_offset(), m.weight.storage_offset())
    test_case.assertEqual(m.weight_orig.size(), m.weight.size())
    test_case.assertEqual(m.weight_orig.stride(), m.weight.stride())

    m = flow.nn.utils.remove_spectral_norm(m)
    test_case.assertFalse(hasattr(m, "weight_orig"))
    test_case.assertFalse(hasattr(m, "weight_u"))

    test_case.assertTrue(hasattr(m, "weight"))
    test_case.assertTrue("weight" in m._parameters)

    with test_case.assertRaisesRegex(RuntimeError, "register two spectral_norm hooks"):
        m = flow.nn.utils.spectral_norm(m)
        m = flow.nn.utils.spectral_norm(m)


def _test_spectral_norm_dim(test_case, device):
    input = flow.randn(2, 3, 10, 12).to(device)
    m = nn.ConvTranspose2d(3, 4, (5, 6)).to(device)
    m = flow.nn.utils.spectral_norm(m)
    x = m(input)
    test_case.assertEqual(m.weight_u.shape, m.weight_orig[0, :, 0, 0].shape)


def _test_spectral_norm_forward(test_case, device):
    k = np.random.randint(1, 100)
    input = flow.randn(21, k).to(device)
    m = nn.Linear(k, 10).to(device)
    m = nn.utils.spectral_norm(m)

    _weight, _bias, _u = m.weight_orig, m.bias, m.weight_u
    _weight_mat = _weight.view(_weight.size(0), -1)
    _v = flow.mv(_weight_mat.t(), _u)
    _v = F.normalize(_v, dim=0, eps=1e-12)
    _u = flow.mv(_weight_mat, _v)
    _u = F.normalize(_u, dim=0, eps=1e-12)
    _weight.data /= flow.dot(_u, flow.matmul(_weight_mat, _v))

    out_hat = F.linear(input, _weight, _bias)
    expect_out = m(input)
    test_case.assertTrue(
        np.allclose(expect_out.numpy(), out_hat.numpy(), rtol=1e-5, atol=1e-5)
    )


@flow.unittest.skip_unless_1n1d()
class TestSpectralNorm(flow.unittest.TestCase):
    @autotest()
    def test_spectral_norm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_spectral_norm,
            _test_spectral_norm_dim,
            _test_spectral_norm_forward,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()

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

import oneflow as flow
import oneflow.unittest
import numpy as np

from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *


def _test_spectral_norm(test_case, device):
    model_flow = flow.nn.Linear(5, 7)
    model_flow = model_flow.to(device)

    m_flow = flow.nn.utils.spectral_norm(model_flow)
    test_case.assertEqual(m_flow.weight_u.size(), flow.Size([m_flow.weight.size(0)]))

    # weight_orig should be trainable
    test_case.assertTrue(hasattr(m_flow, "weight_orig"))
    test_case.assertTrue("weight_orig" in m_flow._parameters)
    # weight_u should be just a reused buffer
    test_case.assertTrue(hasattr(m_flow, "weight_u"))
    test_case.assertTrue("weight_u" in m_flow._buffers)
    test_case.assertTrue("weight_v" in m_flow._buffers)
    # weight should be a plain attribute, not counted as a buffer or a param
    test_case.assertFalse("weight" in m_flow._buffers)
    test_case.assertFalse("weight" in m_flow._parameters)

    test_case.assertEqual(m_flow.weight_orig.size(), m_flow.weight.size())
    test_case.assertEqual(m_flow.weight_orig.stride(), m_flow.weight.stride())

    m_flow = flow.nn.utils.remove_spectral_norm(m_flow)
    test_case.assertFalse(hasattr(m_flow, "weight_orig"))
    test_case.assertFalse(hasattr(m_flow, "weight_u"))
    # weight should be converted back as a parameter
    test_case.assertTrue(hasattr(m_flow, "weight"))
    test_case.assertTrue("weight" in m_flow._parameters)

    with test_case.assertRaisesRegex(RuntimeError, "register two spectral_norm hooks"):
        m_flow = flow.nn.utils.spectral_norm(m_flow)
        m_flow = flow.nn.utils.spectral_norm(m_flow)

    # test correctness in training/eval modes and cpu/multi-gpu settings
    for apply_dp in (True, False):
        if apply_dp:

            def maybe_wrap(m):
                return flow.nn.parallel.DistributedDataParallel(m)

        else:

            def maybe_wrap(m):
                return m

        for requires_grad in (True, False):
            m_flow = flow.nn.Linear(3, 4).to(device)
            m_flow.weight.requires_grad_(requires_grad)
            m_flow = flow.nn.utils.spectral_norm(m_flow)
            wrapped_m = maybe_wrap(m_flow)
            test_case.assertTrue(hasattr(m_flow, "weight_u"))
            u0 = m_flow.weight_u.clone()
            v0 = m_flow.weight_v.clone()

            # TEST TRAINING BEHAVIOR

            # assert that u and v are updated
            input_tensor = flow.randn(2, 3, device=device)
            out = wrapped_m(input_tensor)
            test_case.assertFalse(np.allclose(u0.numpy(), m_flow.weight_u.numpy()))
            test_case.assertFalse(np.allclose(v0.numpy(), m_flow.weight_v.numpy()))

            if requires_grad:
                flow.autograd.grad(out.sum(), m_flow.weight_orig)
            # test removing
            pre_remove_out = wrapped_m(input_tensor)
            m_flow = flow.nn.utils.remove_spectral_norm(m_flow)
            test_case.assertTrue(
                np.allclose(wrapped_m(input_tensor).numpy(), pre_remove_out.numpy())
            )

            m_flow = flow.nn.utils.spectral_norm(m_flow)
            for _ in range(3):
                pre_remove_out = wrapped_m(input_tensor)
            m_flow = flow.nn.utils.remove_spectral_norm(m_flow)
            test_case.assertTrue(
                np.allclose(wrapped_m(input_tensor).numpy(), pre_remove_out.numpy())
            )

            # TEST EVAL BEHAVIOR

            m_flow = flow.nn.utils.spectral_norm(m_flow)
            wrapped_m(input_tensor)
            last_train_out = wrapped_m(input_tensor)
            last_train_u = m_flow.weight_u.clone()
            last_train_v = m_flow.weight_v.clone()
            wrapped_m.zero_grad()
            wrapped_m.eval()

            eval_out0 = wrapped_m(input_tensor)
            # assert eval gives same result as last training iteration
            test_case.assertTrue(np.allclose(eval_out0.numpy(), last_train_out.numpy()))
            # assert doing more iteartion in eval don't change things
            test_case.assertTrue(
                np.allclose(wrapped_m(input_tensor).numpy(), eval_out0.numpy())
            )
            test_case.assertTrue(
                np.allclose(last_train_u.numpy(), m_flow.weight_u.numpy())
            )
            test_case.assertTrue(
                np.allclose(last_train_v.numpy(), m_flow.weight_v.numpy())
            )


def _test_spectral_norm_dim(test_case, device):
    inp = flow.randn(2, 3, 10, 12)
    m_flow = flow.nn.ConvTranspose2d(3, 4, (5, 6))
    m_flow = flow.nn.utils.spectral_norm(m_flow)

    # this should not run into incompatible shapes
    x = m_flow(inp)

    # check that u refers to the same dimension
    test_case.assertEqual(m_flow.weight_u.shape, m_flow.weight_orig[0, :, 0, 0].shape)


def _test_spectral_norm_forward(test_case, device):
    input_tensor = flow.randn(3, 5)
    m_flow = flow.nn.Linear(5, 7)
    m_flow = flow.nn.utils.spectral_norm(m_flow)
    # naive forward

    _weight, _bias, _u = m_flow.weight_orig, m_flow.bias, m_flow.weight_u
    _weight_mat = _weight.view(_weight.size(0), -1)
    _v = flow.mv(_weight_mat.t(), _u)

    _v = flow.nn.functional.normalize(_v, dim=0, eps=1e-12)
    _u = flow.mv(_weight_mat, _v)
    _u = flow.nn.functional.normalize(_u, dim=0, eps=1e-12)

    _weight.data /= flow.dot(_u, flow.matmul(_weight_mat, _v))
    out_hat = flow.nn.functional.linear(input_tensor, _weight, _bias)
    expect_out = m_flow(input_tensor)
    test_case.assertTrue(np.allclose(expect_out.numpy(), out_hat.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestSpectralNorm(flow.unittest.TestCase):
    def test_spectralnorm(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_spectral_norm_dim,
            _test_spectral_norm_forward,
            _test_spectral_norm,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=10, auto_backward=False, check_graph="ValidatedFalse")
    def test_spectral_norm_with_random_data(test_case):
        device = random_device()

        output = random(2, 6).to(int)
        input = random(2, 6).to(int)

        model_torch = torch.nn.Linear(output, input)
        model_torch = model_torch.to(device)
        m = torch.nn.utils.spectral_norm(model_torch)
        return m.weight_u, m.weight_v


if __name__ == "__main__":
    unittest.main()

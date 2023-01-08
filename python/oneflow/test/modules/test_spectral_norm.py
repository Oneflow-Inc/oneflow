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
from copy import deepcopy

import numpy as np
import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F
import oneflow.unittest

from oneflow.test_utils.test_util import GenArgList

import torch as torch_original
from oneflow.test_utils.automated_test_util import *


def maybe_wrap(m):
    return m


def _test_spectral_norm_impl(test_case, device):
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


def _test_spectral_norm_training_update_and_remove(test_case, device):
    for requires_grad in [True, False]:
        m = nn.Linear(5, 7).to(device)
        m.weight.requires_grad_(requires_grad)
        m = nn.utils.spectral_norm(m)
        wrapped_m = maybe_wrap(m)

        test_case.assertTrue(hasattr(m, "weight_u"))
        u0 = m.weight_u.clone()
        v0 = m.weight_v.clone()

        input = flow.rand((3, 5)).to(device)
        out = wrapped_m(input)
        test_case.assertFalse(
            np.allclose(
                u0.cpu().detach().numpy(), 
                m.weight_u.cpu().detach().numpy()
            )
        )
        test_case.assertFalse(
            np.allclose(
                v0.cpu().detach().numpy(), 
                m.weight_v.cpu().detach().numpy()
            )
        )

        if requires_grad:
            flow.autograd.grad(out.sum(), m.weight_orig)

        saved_u = m.weight_u.clone()
        saved_v = m.weight_v.clone()

        # need grad check

        # remove
        pre_remove_out = wrapped_m(input)
        m = nn.utils.remove_spectral_norm(m)
        test_case.assertTrue(
            np.allclose(
                wrapped_m(input).cpu().detach().numpy(), 
                pre_remove_out.cpu().detach().numpy()
            )
        )

        m = nn.utils.spectral_norm(m)
        for _ in range(3):
            pre_remove_out = wrapped_m(input)
        m = nn.utils.remove_spectral_norm(m)
        test_case.assertTrue(
            np.allclose(
                wrapped_m(input).cpu().detach().numpy(), 
                pre_remove_out.cpu().detach().numpy()
            )
        )


def _test_spectral_norm_eval_update(test_case, device):
    m = nn.Linear(5, 7).to(device)
    m = nn.utils.spectral_norm(m)
    wrapped_m = maybe_wrap(m)
    input = flow.rand((3, 5)).to(device)
    last_train_out = wrapped_m(input)
    last_train_u = m.weight_u.clone()
    last_train_v = m.weight_v.clone()

    wrapped_m.zero_grad()
    wrapped_m.eval()

    eval_out0 = wrapped_m(input)
    test_case.assertTrue(
        np.allclose(
            eval_out0.cpu().detach().numpy(), 
            last_train_out.cpu().detach().numpy()
        )
    )
    test_case.assertTrue(
        np.allclose(
            eval_out0.cpu().detach().numpy(), 
            wrapped_m(input).cpu().detach().numpy()
        )
    )
    test_case.assertTrue(
        np.allclose(
            eval_out0.cpu().detach().numpy(), 
            wrapped_m(input).cpu().detach().numpy()
        )
    )
    test_case.assertTrue(
        np.allclose(
            last_train_u.cpu().detach().numpy(), 
            m.weight_u.cpu().detach().numpy()
        )
    )
    test_case.assertTrue(
        np.allclose(
            last_train_v.cpu().detach().numpy(), 
            m.weight_v.cpu().detach().numpy()
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestSpectralNorm(flow.unittest.TestCase):
    # @autotest(check_graph="ValidatedFalse")
    # def test_spectral_norm_with_random_data(test_case):
    #     device = random_device()

    #     input = random(1, 10).to(int)
    #     output = random(1, 10).to(int)

    #     model_torch = torch.nn.Linear(output, input).to(device)
    #     m = torch.nn.utils.spectral_norm(model_torch)
    #     # only weight_orig should be setattr "requires_grad=True"
    #     return m.weight_orig

    # @autotest()
    # def test_spectral_norm(test_case):
    #     arg_dict = OrderedDict()
    #     arg_dict["test_fun"] = [
    #         _test_spectral_norm_impl,
    #         _test_spectral_norm_dim,
    #         _test_spectral_norm_forward,
    #         _test_spectral_norm_training_update_and_remove,
    #         _test_spectral_norm_eval_update,
    #     ]
    #     arg_dict["device"] = ["cpu", "cuda"]
    #     for arg in GenArgList(arg_dict):
    #         arg[0](test_case, *arg[1:])
    
    @autotest()
    def test_spectral_norm_load_state_dict(test_case):
        inp = flow.randn(2, 3)
        for activate_times in (0, 3):
            # Test backward compatibility
            # At version None -> 1: weight becomes not a buffer and v vector becomes a buffer
            m = nn.Linear(3, 5)
            snm = nn.utils.spectral_norm(m)
            snm.train()
            for _ in range(activate_times):
                snm(inp)

            version_latest_ref_state_dict = deepcopy(snm.state_dict())
            test_case.assertEqual({'weight_orig', 'bias', 'weight_u', 'weight_v'}, set(version_latest_ref_state_dict.keys()))

            # test that non-strict loading works
            non_strict_state_dict = deepcopy(version_latest_ref_state_dict)
            non_strict_state_dict['nonsense'] = 'nonsense'
            with test_case.assertRaisesRegex(RuntimeError, r'Unexpected key\(s\) in state_dict: "nonsense"'):
                snm.load_state_dict(non_strict_state_dict, strict=True)
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_orig']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_u']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_v']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            non_strict_state_dict['weight'] = snm.weight.detach().clone()  # set W as a buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            # del non_strict_state_dict._metadata['']['spectral_norm']       # remove metadata info
            # snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight']                            # remove W buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['bias']
            snm.load_state_dict(non_strict_state_dict, strict=False)

            # craft a version None state_dict
            version_none_state_dict = deepcopy(version_latest_ref_state_dict)
            # test_case.assertIn('spectral_norm', version_none_state_dict._metadata[''])
            # del version_none_state_dict._metadata['']['spectral_norm']       # remove metadata info
            del version_none_state_dict['weight_v']                          # remove v vector
            version_none_state_dict['weight'] = snm.weight.detach().clone()  # set W as a buffer

            # normal state_dict
            for version_latest_with_metadata in [True, False]:
                version_latest_state_dict = deepcopy(version_latest_ref_state_dict)

                # if not version_latest_with_metadata:
                #     # We want to still load a user-crafted state_dict, one without metadata
                #     del version_latest_state_dict._metadata['']['spectral_norm']

                # test that re-wrapping does not matter
                m = nn.utils.remove_spectral_norm(snm)
                snm = nn.utils.spectral_norm(m)

                snm.load_state_dict(version_latest_ref_state_dict)
                with flow.no_grad():
                    snm.eval()
                    out0_eval = snm(inp)
                    snm.train()
                    out1_train = snm(inp)
                    out2_train = snm(inp)
                    snm.eval()
                    out3_eval = snm(inp)

                # test that re-wrapping does not matter
                m = nn.utils.remove_spectral_norm(snm)
                snm = nn.utils.spectral_norm(m)

                snm.load_state_dict(version_none_state_dict)
                # if activate_times > 0:
                #     # since in loading version None state dict, we assume that the
                #     # values in the state dict have gone through at lease one
                #     # forward, we only test for equivalence when activate_times > 0.
                #     with flow.no_grad():
                #         snm.eval()
                #         test_case.assertEqual(out0_eval, snm(inp))
                #         snm.train()
                #         test_case.assertEqual(out1_train, snm(inp))
                #         test_case.assertEqual(out2_train, snm(inp))
                #         snm.eval()
                #         test_case.assertEqual(out3_eval, snm(inp))

                # # test that re-wrapping does not matter
                # m = nn.utils.remove_spectral_norm(snm)
                # snm = nn.utils.spectral_norm(m)

                # # Test normal loading
                # snm.load_state_dict(version_latest_state_dict)
                # with flow.no_grad():
                #     snm.eval()
                #     test_case.assertEqual(out0_eval, snm(inp))
                #     snm.train()
                #     test_case.assertEqual(out1_train, snm(inp))
                #     test_case.assertEqual(out2_train, snm(inp))
                #     snm.eval()
                #     test_case.assertEqual(out3_eval, snm(inp))


if __name__ == "__main__":
    unittest.main()

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
import os
import unittest
import numpy as np
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestBfloat16Activatian(flow.unittest.TestCase):
    def test_tan_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.tan(x)
        fp32_y = flow.tan(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_tanh_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.tanh(x)
        fp32_y = flow.tanh(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_sin_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.sin(x)
        fp32_y = flow.sin(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_sinh_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.sinh(x)
        fp32_y = flow.sinh(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_cos_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.cos(x)
        fp32_y = flow.cos(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_cosh_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.cosh(x)
        fp32_y = flow.cosh(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_atan_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.atan(x)
        fp32_y = flow.atan(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_atanh_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.atanh(x)
        fp32_y = flow.atanh(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_asin_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.asin(x)
        fp32_y = flow.asin(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_asinh_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.asinh(x)
        fp32_y = flow.asinh(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_acos_with_random_data(test_case):
        np_array = np.random.uniform(-1, 1, (4, 4))
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.acos(x)
        fp32_y = flow.acos(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_acosh_with_random_data(test_case):
        np_array = np.random.uniform(1, 5, (4, 4))
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.acosh(x)
        fp32_y = flow.acosh(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_sqrt_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.sqrt(x)
        fp32_y = flow.sqrt(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_square_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.square(x)
        fp32_y = flow.square(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_exp_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.exp(x)
        fp32_y = flow.exp(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_exp2_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.exp2(x)
        fp32_y = flow.exp2(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_ceil_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.ceil(x)
        fp32_y = flow.ceil(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_erf_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.erf(x)
        fp32_y = flow.erf(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_erfc_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.erfc(x)
        fp32_y = flow.erfc(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_floor_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.floor(x)
        fp32_y = flow.floor(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_expm1_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.expm1(x)
        fp32_y = flow.expm1(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_lgamma_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.lgamma(x)
        fp32_y = flow.lgamma(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_log_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.log(x)
        fp32_y = flow.log(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_log2_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.log2(x)
        fp32_y = flow.log2(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_log1p_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.log1p(x)
        fp32_y = flow.log1p(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_sigmoid_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.sigmoid(x)
        fp32_y = flow.sigmoid(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_round_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.round(x)
        fp32_y = flow.round(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_rsqrt_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.rsqrt(x)
        fp32_y = flow.rsqrt(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_softplus_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.softplus(x)
        fp32_y = flow.softplus(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_softsign_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.softsign(x)
        fp32_y = flow.softsign(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_softshrink_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.softshrink(x)
        fp32_y = flow.softshrink(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_silu_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.silu(x)
        fp32_y = flow.silu(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_selu_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.selu(x)
        fp32_y = flow.selu(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_mish_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.mish(x)
        fp32_y = flow.mish(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_gelu_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        y = flow.gelu(x)
        fp32_y = flow.gelu(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_elu_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        elu = flow.nn.ELU()
        y = elu(x)
        fp32_y = elu(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_celu_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        celu = flow.nn.CELU()
        y = celu(x)
        fp32_y = celu(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_hardswish_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        hardswish = flow.nn.Hardswish()
        y = hardswish(x)
        fp32_y = hardswish(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_hardswish_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        hardsigmoid = flow.nn.Hardsigmoid()
        y = hardsigmoid(x)
        fp32_y = hardsigmoid(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_hardshrink_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        hardshrink = flow.nn.Hardshrink()
        y = hardshrink(x)
        fp32_y = hardshrink(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_hardtanh_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        hardtanh = flow.nn.Hardtanh()
        y = hardtanh(x)
        fp32_y = hardtanh(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_leakyrelu_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        leakyrelu = flow.nn.LeakyReLU(0.1)
        y = leakyrelu(x)
        fp32_y = leakyrelu(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_threshold_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        th = flow.nn.Threshold(threshold=0.5, value=0.2)
        y = th(x)
        fp32_y = th(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )

    def test_logsinmoid_with_random_data(test_case):
        np_array = np.random.rand(4, 4)
        x = flow.tensor(np_array, dtype=flow.bfloat16, device="cpu")
        fp32_x = x.float()
        logsigmoid = flow.nn.LogSigmoid()
        y = logsigmoid(x)
        fp32_y = logsigmoid(fp32_x)
        test_case.assertTrue(
            np.allclose(
                y.float().numpy(),
                fp32_y.bfloat16().float().numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )


if __name__ == "__main__":
    unittest.main()

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


class TestGenerator(flow.unittest.TestCase):
    def test_different_devices(test_case):
        auto_gen = flow.Generator(device="auto")
        cpu_gen = flow.Generator(device="cpu")
        test_case.assertTrue(auto_gen.initial_seed() == cpu_gen.initial_seed())
        with test_case.assertRaises(RuntimeError) as context:
            flow.Generator(device="invalid")
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            cuda_gen = flow.Generator(device="cuda")
            test_case.assertTrue(auto_gen.initial_seed() == cuda_gen.initial_seed())

    def test_generator_manual_seed(test_case):
        generator = flow.Generator()
        generator.manual_seed(1)
        test_case.assertTrue(generator.initial_seed() == 1)
        generator.manual_seed(2)
        test_case.assertTrue(generator.initial_seed() == 2)

    def test_generator_in_dropout(test_case):
        tgt = flow.ones(2000000)
        output = flow._C.dropout(
            tgt, p=0.1, training=True, generator=flow.Generator(), addend=None
        )
        output.numpy()
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            output = flow._C.dropout(
                tgt.cuda(), 0.1, training=True, generator=flow.Generator(), addend=None
            )
            output.numpy()


class TestDefaultGenerator(flow.unittest.TestCase):
    def test_different_devices(test_case):
        auto_gen = flow.Generator(device="auto")
        cpu_gen = flow.default_generator
        with test_case.assertRaises(RuntimeError) as context:
            flow.Generator(device="invalid")

        flow.Generator(device="cpu:1000")
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            with test_case.assertRaises(
                oneflow._oneflow_internal.exception.Exception
            ) as context:
                flow.Generator(device="cuda:1000")
            cuda_gen = flow.Generator(device="cuda")
            cuda0_gen = flow.Generator(device="cuda:0")

    def test_generator_manual_seed(test_case):
        cpu_gen = flow.default_generator
        auto_gen = flow.Generator(device="auto")
        test_gens = [cpu_gen, auto_gen]
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            cuda_gen = flow.Generator(device="cuda")
            cuda0_gen = flow.Generator(device="cuda:0")
            test_gens += [cuda_gen, cuda0_gen]
        for seed in [1, 2]:
            for gen in test_gens:
                gen.manual_seed(seed)
                test_case.assertTrue(gen.initial_seed() == seed)

    def test_generator_seed(test_case):
        cpu_gen = flow.default_generator
        auto_gen = flow.Generator(device="auto")
        test_gens = [auto_gen, cpu_gen]
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            cuda_gen = flow.Generator(device="cuda")
            cuda0_gen = flow.Generator(device="cuda:0")
            test_gens += [cuda_gen, cuda0_gen]
        for gen in test_gens:
            seed = gen.seed()
            test_case.assertTrue(seed == gen.initial_seed())

    def test_generator_getstate(test_case):
        auto_gen = flow.Generator(device="auto")
        state = auto_gen.get_state()
        cpu_gen = flow.Generator(device="cpu")
        state = cpu_gen.get_state()
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            cuda_gen = flow.Generator(device="cuda")
            state = cuda_gen.get_state()

    @unittest.skip("the curandstate is no longer used by normal kernel")
    def test_generator_setstate(test_case):
        cpu_gen = flow.default_generator
        flow.randn(100, 100, dtype=flow.float32, device="cpu", generator=cpu_gen)
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            cuda_gen = flow.Generator("cuda")
            flow.randn(100, 100, dtype=flow.float32, device="cuda", generator=cuda_gen)
        state = cpu_gen.get_state()
        flow.randn(100, 100, dtype=flow.float32, device="cpu", generator=cpu_gen)
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            cuda_state = cuda_gen.get_state()
            flow.randn(100, 100, dtype=flow.float32, device="cuda", generator=cuda_gen)

        new_state = cpu_gen.get_state()
        test_case.assertTrue(not np.allclose(new_state.numpy(), state.numpy()))

        cpu_gen.set_state(state)
        new_state = cpu_gen.get_state()
        test_case.assertTrue(np.allclose(new_state.numpy(), state.numpy()))

        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            new_cuda_state = cuda_gen.get_state()
            test_case.assertTrue(
                not np.allclose(new_cuda_state.numpy(), cuda_state.numpy())
            )

            cuda_gen.set_state(cuda_state)
            new_cuda_state = cuda_gen.get_state()
            test_case.assertTrue(
                np.allclose(new_cuda_state.numpy(), cuda_state.numpy())
            )

    def test_get_rng_state(test_case):
        cpu_gen = flow.default_generator
        state = cpu_gen.get_state()
        rng_state = flow.get_rng_state()
        test_case.assertTrue(np.allclose(state.numpy(), rng_state.numpy()))

        flow.randn(100, 100, dtype=flow.float32, device="cpu", generator=cpu_gen)
        state = cpu_gen.get_state()
        rng_state = flow.get_rng_state()
        test_case.assertTrue(np.allclose(state.numpy(), rng_state.numpy()))

    def test_set_rng_state(test_case):
        flow.randn(100, 100)
        state = flow.get_rng_state()
        flow.randn(100, 100)

        new_state = flow.get_rng_state()
        test_case.assertTrue(not np.allclose(new_state.numpy(), state.numpy()))

        flow.set_rng_state(state)
        new_state = flow.get_rng_state()
        test_case.assertTrue(np.allclose(new_state.numpy(), state.numpy()))

    # NOTE: according to https://github.com/Oneflow-Inc/oneflow/pull/9102#discussion_r973811389
    # tensor init function fallback to `flow.default_generator.seed()`, and this test will be normal while tensor init functions reconstructed.(using op/kernel)
    # @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @unittest.skipIf(True, "tensor init functions need to be reconstructed!")
    def test_tensor_init(test_case):
        flow.manual_seed(0)
        x = flow.ones(2)
        x.uniform_()

        flow.manual_seed(0)
        y = flow.ones(2).to("cuda")
        y.uniform_()

        test_case.assertTrue(np.allclose(x.numpy(), y.numpy()))


if __name__ == "__main__":
    unittest.main()

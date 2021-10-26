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
        with test_case.assertRaises(
            oneflow._oneflow_internal.exception.RuntimeException
        ) as context:
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
        output = flow._C.dropout(tgt, 0.1, True, flow.Generator())
        output.numpy()
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            output = flow._C.dropout(tgt.cuda(), 0.1, True, flow.Generator())
            output.numpy()


class TestDefaultGenerator(flow.unittest.TestCase):
    def test_global_manual_seed(test_case):
        global_seed = 10
        flow.manual_seed(10)
        auto_gen = flow.default_generator(device="auto")
        cpu_gen = flow.default_generator(device="cpu")
        test_gens = [auto_gen, cpu_gen]
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            cuda_gen = flow.default_generator(device="cuda")
            cuda0_gen = flow.default_generator(device="cuda:0")
            test_gens += [cuda_gen, cuda0_gen]
        for gen in test_gens:
            test_case.assertTrue(gen.initial_seed() == global_seed)

    def test_different_devices(test_case):
        auto_gen = flow.default_generator(device="auto")
        cpu_gen = flow.default_generator(device="cpu")
        with test_case.assertRaises(
            oneflow._oneflow_internal.exception.RuntimeException
        ) as context:
            flow.default_generator(device="invalid")

        flow.default_generator(device="cpu:1000")
        test_gens = [cpu_gen]
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            with test_case.assertRaises(
                oneflow._oneflow_internal.exception.CheckFailedException
            ) as context:
                flow.default_generator(device="cuda:1000")
            cuda_gen = flow.default_generator(device="cuda")
            cuda0_gen = flow.default_generator(device="cuda:0")
            test_gens += [cuda_gen, cuda0_gen]
        for gen in test_gens:
            test_case.assertTrue(auto_gen.initial_seed() == gen.initial_seed())

    def test_generator_manual_seed(test_case):
        auto_gen = flow.default_generator(device="auto")
        cpu_gen = flow.default_generator(device="cpu")
        test_gens = [auto_gen, cpu_gen]
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            cuda_gen = flow.default_generator(device="cuda")
            cuda0_gen = flow.default_generator(device="cuda:0")
            test_gens += [cuda_gen, cuda0_gen]
        for seed in [1, 2]:
            auto_gen.manual_seed(seed)
            for gen in test_gens:
                test_case.assertTrue(gen.initial_seed() == seed)

    def test_generator_seed(test_case):
        auto_gen = flow.default_generator(device="auto")
        cpu_gen = flow.default_generator(device="cpu")
        test_gens = [auto_gen, cpu_gen]
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            cuda_gen = flow.default_generator(device="cuda")
            cuda0_gen = flow.default_generator(device="cuda:0")
            test_gens += [cuda_gen, cuda0_gen]
        for gen in test_gens:
            seed = gen.seed()
            test_case.assertTrue(seed == gen.initial_seed())

    def test_generator_getstate(test_case):
        auto_gen = flow.Generator(device="auto")
        state = auto_gen.get_state()
        cpu_gen = flow.Generator(device="cpu")
        state = cpu_gen.get_state()
        auto_gen = flow.default_generator(device="auto")
        state = auto_gen.get_state()
        cpu_gen = flow.default_generator(device="cpu")
        state = cpu_gen.get_state()
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            cuda_gen = flow.Generator(device="cuda")
            state = cuda_gen.get_state()
            cuda_gen = flow.default_generator(device="cuda")
            state = cuda_gen.get_state()

    def test_generator_setstate(test_case):
        auto_gen = flow.default_generator(device="auto")
        flow.randn(100, 100, dtype=flow.float32, device="cpu", generator=auto_gen)
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            flow.randn(100, 100, dtype=flow.float32, device="cuda", generator=auto_gen)
        state = auto_gen.get_state()
        flow.randn(100, 100, dtype=flow.float32, device="cpu", generator=auto_gen)
        if not os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            flow.randn(100, 100, dtype=flow.float32, device="cuda", generator=auto_gen)

        new_state = auto_gen.get_state()
        test_case.assertTrue(not np.allclose(new_state.numpy(), state.numpy()))

        auto_gen.set_state(state)
        new_state = auto_gen.get_state()
        test_case.assertTrue(np.allclose(new_state.numpy(), state.numpy()))


if __name__ == "__main__":
    unittest.main()

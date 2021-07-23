import os
import unittest
import oneflow as flow

class TestGenerator(flow.unittest.TestCase):

    def test_different_devices(test_case):
        auto_gen = flow.Generator(device='auto')
        cpu_gen = flow.Generator(device='cpu')
        test_case.assertTrue(auto_gen.initial_seed(), cpu_gen.initial_seed())
        with test_case.assertRaises(Exception) as context:
            flow.Generator(device='invalid')
        test_case.assertTrue('unimplemented' in str(context.exception))
        if not os.getenv('ONEFLOW_TEST_CPU_ONLY'):
            cuda_gen = flow.Generator(device='cuda')
            test_case.assertTrue(auto_gen.initial_seed(), cuda_gen.initial_seed())

    def test_generator_manual_seed(test_case):
        generator = flow.Generator()
        generator.manual_seed(1)
        test_case.assertTrue(generator.initial_seed() == 1)
        generator.manual_seed(2)
        test_case.assertTrue(generator.initial_seed() == 2)

class TestDefaultGenerator(flow.unittest.TestCase):

    def test_global_manual_seed(test_case):
        global_seed = 10
        flow.manual_seed(10)
        auto_gen = flow.default_generator(device='auto')
        cpu_gen = flow.default_generator(device='cpu')
        test_gens = [auto_gen, cpu_gen]
        if not os.getenv('ONEFLOW_TEST_CPU_ONLY'):
            cuda_gen = flow.default_generator(device='cuda')
            cuda0_gen = flow.default_generator(device='cuda:0')
            test_gens += [cuda_gen, cuda0_gen]
        for gen in test_gens:
            test_case.assertTrue(gen.initial_seed() == global_seed)

    def test_different_devices(test_case):
        auto_gen = flow.default_generator(device='auto')
        cpu_gen = flow.default_generator(device='cpu')
        with test_case.assertRaises(Exception) as context:
            flow.default_generator(device='invalid')
        test_case.assertTrue('unimplemented' in str(context.exception))
        with test_case.assertRaises(Exception) as context:
            flow.default_generator(device='cpu:1000')
        test_case.assertTrue('check_failed' in str(context.exception))
        test_gens = [cpu_gen]
        if not os.getenv('ONEFLOW_TEST_CPU_ONLY'):
            with test_case.assertRaises(Exception) as context:
                flow.default_generator(device='cuda:1000')
            test_case.assertTrue('check_failed' in str(context.exception))
            cuda_gen = flow.default_generator(device='cuda')
            cuda0_gen = flow.default_generator(device='cuda:0')
            test_gens += [cuda_gen, cuda0_gen]
        for gen in test_gens:
            test_case.assertTrue(auto_gen.initial_seed() == gen.initial_seed())

    def test_generator_manual_seed(test_case):
        auto_gen = flow.default_generator(device='auto')
        cpu_gen = flow.default_generator(device='cpu')
        test_gens = [auto_gen, cpu_gen]
        if not os.getenv('ONEFLOW_TEST_CPU_ONLY'):
            cuda_gen = flow.default_generator(device='cuda')
            cuda0_gen = flow.default_generator(device='cuda:0')
            test_gens += [cuda_gen, cuda0_gen]
        for seed in [1, 2]:
            auto_gen.manual_seed(seed)
            for gen in test_gens:
                test_case.assertTrue(gen.initial_seed() == seed)

    def test_generator_seed(test_case):
        auto_gen = flow.default_generator(device='auto')
        cpu_gen = flow.default_generator(device='cpu')
        test_gens = [auto_gen, cpu_gen]
        if not os.getenv('ONEFLOW_TEST_CPU_ONLY'):
            cuda_gen = flow.default_generator(device='cuda')
            cuda0_gen = flow.default_generator(device='cuda:0')
            test_gens += [cuda_gen, cuda0_gen]
        for gen in test_gens:
            seed = gen.seed()
            test_case.assertTrue(seed == gen.initial_seed())
if __name__ == '__main__':
    unittest.main()
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
import oneflow as flow


class TestGenerator(flow.unittest.TestCase):
    def test_different_devices(test_case):
        auto_gen = flow.Generator(device="auto")
        cuda_gen = flow.Generator(device="cuda")
        cpu_gen = flow.Generator(device="cpu")
        test_case.assertTrue(auto_gen.initial_seed(), cuda_gen.initial_seed())
        test_case.assertTrue(auto_gen.initial_seed(), cpu_gen.initial_seed())
        with test_case.assertRaises(Exception) as context:
            flow.Generator(device="invalid")
        test_case.assertTrue("unimplemented" in str(context.exception))

    def test_global_manual_seed(test_case):
        flow.manual_seed(10)

    def test_generator_manual_seed(test_case):
        generator = flow.Generator()
        generator.manual_seed(1)
        test_case.assertTrue(generator.initial_seed() == 1)
        generator.manual_seed(2)
        test_case.assertTrue(generator.initial_seed() == 2)


if __name__ == "__main__":
    unittest.main()

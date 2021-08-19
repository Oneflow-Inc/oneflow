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

import oneflow.unittest


class TestAutogradMode(oneflow.unittest.TestCase):
    def test_grad_mode(test_case):
        test_case.assertTrue(flow.autograd.grad_mode())

    def test_inference_mode(test_case):
        with flow.inference_mode(True):
            test_case.assertFalse(flow.autograd.grad_mode())

        @flow.inference_mode(True)
        def func():
            test_case.assertFalse(flow.autograd.grad_mode())

        func()

        with flow.inference_mode(False):
            test_case.assertTrue(flow.autograd.grad_mode())

        @flow.inference_mode(False)
        def func():
            test_case.assertTrue(flow.autograd.grad_mode())

        func()

    def test_grad_enable(test_case):
        with flow.grad_enable():
            test_case.assertTrue(flow.autograd.grad_mode())

        @flow.grad_enable()
        def func():
            test_case.assertTrue(flow.autograd.grad_mode())

        func()

    def test_no_grad(test_case):
        with flow.no_grad():
            test_case.assertFalse(flow.autograd.grad_mode())

        @flow.no_grad()
        def func():
            test_case.assertFalse(flow.autograd.grad_mode())

        func()


if __name__ == "__main__":
    unittest.main()

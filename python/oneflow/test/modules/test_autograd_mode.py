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
        test_case.assertTrue(flow.is_grad_enabled())

    def test_inference_mode(test_case):
        with flow.inference_mode(True):
            test_case.assertFalse(flow.is_grad_enabled())
        test_case.assertTrue(flow.is_grad_enabled())

        @flow.inference_mode(True)
        def func():
            test_case.assertFalse(flow.is_grad_enabled())

        func()
        test_case.assertTrue(flow.is_grad_enabled())

        with flow.inference_mode(False):
            test_case.assertTrue(flow.is_grad_enabled())
        test_case.assertTrue(flow.is_grad_enabled())

        @flow.inference_mode(False)
        def func():
            test_case.assertTrue(flow.is_grad_enabled())

        func()
        test_case.assertTrue(flow.is_grad_enabled())

    def test_enable_grad(test_case):
        with flow.enable_grad():
            test_case.assertTrue(flow.is_grad_enabled())
        test_case.assertTrue(flow.is_grad_enabled())

        @flow.enable_grad()
        def func():
            test_case.assertTrue(flow.is_grad_enabled())

        func()
        test_case.assertTrue(flow.is_grad_enabled())

    def test_no_grad(test_case):
        with flow.no_grad():
            test_case.assertFalse(flow.is_grad_enabled())
        test_case.assertTrue(flow.is_grad_enabled())

        @flow.no_grad()
        def func():
            test_case.assertFalse(flow.is_grad_enabled())

        func()
        test_case.assertTrue(flow.is_grad_enabled())

    def test_set_grad_enabled(test_case):
        def assert_grad_mode(mode):
            if mode:
                test_case.assertTrue(flow.is_grad_enabled())
            else:
                test_case.assertFalse(flow.is_grad_enabled())

        def get_decorater_func_with_mode(mode):
            @flow.set_grad_enabled(mode)
            def func():
                assert_grad_mode(mode)

            return func

        def get_decorater_context_func_with_mode(dec_mode, ctx_mode):
            @flow.set_grad_enabled(dec_mode)
            def func():
                assert_grad_mode(dec_mode)
                with flow.set_grad_enabled(ctx_mode):
                    assert_grad_mode(ctx_mode)
                assert_grad_mode(dec_mode)

            return func

        flow.set_grad_enabled(False)
        assert_grad_mode(False)

        with flow.set_grad_enabled(True):
            assert_grad_mode(True)
            flow.set_grad_enabled(False)
            assert_grad_mode(False)
            func = get_decorater_func_with_mode(True)
            func()
        assert_grad_mode(False)

        flow.set_grad_enabled(True)
        assert_grad_mode(True)

        with flow.set_grad_enabled(False):
            assert_grad_mode(False)
            flow.set_grad_enabled(True)
            assert_grad_mode(True)
            func = get_decorater_func_with_mode(False)
            func()
        assert_grad_mode(True)

        get_decorater_context_func_with_mode(True, True)()
        get_decorater_context_func_with_mode(True, False)()
        get_decorater_context_func_with_mode(False, True)()
        get_decorater_context_func_with_mode(False, False)()


if __name__ == "__main__":
    unittest.main()

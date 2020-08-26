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
import oneflow as flow
import oneflow.typing as oft


def test_user_op_module_builder_in_namespace(test_case):
    flow.clear_default_session()

    @flow.global_function()
    def foo() -> oft.Numpy:
        with flow.scope.namespace("foo"):
            flip = flow.random.coin_flip(name="CoinFlip")

        test_case.assertTrue(flip.op_name == "foo-CoinFlip")
        return flip

    ret = foo()
    test_case.assertTrue(ret.item() == 0 or ret.item() == 1)

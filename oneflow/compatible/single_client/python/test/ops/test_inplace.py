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
import numpy as np
import oneflow as flow
import oneflow.typing as oft


def MakeFuncConfig(enable_inplace):
    func_config = flow.FunctionConfig()
    func_config.enable_inplace(enable_inplace)
    return func_config


def TrainCompare(test_case, func):
    func_config = MakeFuncConfig(True)

    @flow.global_function(type="train", function_config=func_config)
    def EnableInplace():
        return func("w0")

    func_config.enable_inplace(False)

    @flow.global_function(type="train", function_config=func_config)
    def DisableInplace():
        return func("w1")

    num_iter = 10
    enable_inplace_losses = np.array(
        [EnableInplace().get().tolist() for _ in range(num_iter)]
    )
    disable_inplace_losses = np.array(
        [DisableInplace().get().tolist() for _ in range(num_iter)]
    )
    test_case.assertTrue(np.allclose(enable_inplace_losses, disable_inplace_losses))


@flow.unittest.skip_unless_1n1d()
class TestInplace(flow.unittest.TestCase):
    def test_loss_inplace(test_case):
        def IdentityLoss(name):
            w = flow.get_variable(
                name, (10,), initializer=flow.constant_initializer(100)
            )
            y = flow.math.reduce_sum(w)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [5]), momentum=0
            ).minimize(y)
            return y

        TrainCompare(test_case, IdentityLoss)

    def test_inplace_variable(test_case):
        @flow.global_function(function_config=MakeFuncConfig(True))
        def InplaceVariable():
            w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(1))
            y = flow.math.relu(w)
            return y

        test_case.assertTrue(
            np.allclose(InplaceVariable().get().numpy(), np.ones((10,), np.float32))
        )

    def test_deadlock(test_case):
        @flow.global_function(function_config=MakeFuncConfig(True))
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.relu(x)
            y = flow.math.relu(y)

        Foo(np.ones((10,), dtype=np.float32))

    def test_nodeadlock_with_return(test_case):
        @flow.global_function(function_config=MakeFuncConfig(True))
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.relu(x)
            y = flow.math.relu(y)
            return y

        Foo(np.ones((10,), dtype=np.float32)).get()

    def test_reentrant_lock_check_failed(test_case):
        @flow.global_function(function_config=MakeFuncConfig(True))
        def Foo(x: oft.Numpy.Placeholder((10,))):
            y = flow.math.relu(x)
            y = flow.math.relu(y)

        Foo(np.ones((10,), dtype=np.float32))

    def test_const_inplace_variable(test_case):
        @flow.global_function(function_config=MakeFuncConfig(True))
        def InplaceVariable():
            w = flow.get_variable("w", (2, 5), initializer=flow.constant_initializer(1))
            y = flow.reshape(w, (10,))
            return y

        of_ret = InplaceVariable().get().numpy()
        test_case.assertTrue(np.allclose(of_ret, np.ones((10,), np.float32)))


if __name__ == "__main__":
    unittest.main()

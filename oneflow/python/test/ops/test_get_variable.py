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


@flow.unittest.skip_unless_1n1d()
class TestGetVariable(flow.unittest.TestCase):
    def test_get_variable_with_same_name(test_case):
        flow.clear_default_session()
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)

        def get_v():
            return flow.get_variable(
                name="var",
                shape=(5, 2),
                dtype=flow.float32,
                initializer=flow.random_uniform_initializer(),
            )

        @flow.global_function(function_config=func_config)
        def TestJob0():
            v1 = get_v()
            v2 = get_v()
            return v1, v2

        @flow.global_function(function_config=func_config)
        def TestJob1():
            return get_v()

        j0_v1, j0_v2 = TestJob0().get()
        j1_v = TestJob1().get()
        test_case.assertTrue(np.array_equal(j0_v1.numpy(), j0_v2.numpy()))
        test_case.assertTrue(np.array_equal(j0_v1.numpy(), j1_v.numpy()))

    def test_get_job_shared_variable(test_case):
        flow.clear_default_session()

        def get_var(name, shape=(2, 5), dtype=flow.float, trainable=False):
            return flow.get_variable(
                name=name,
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                initializer=flow.random_uniform_initializer(),
            )

        learning_rate = 1e-2

        @flow.global_function(type="train", function_config=flow.FunctionConfig())
        def train(x_def: oft.Numpy.Placeholder(shape=(2, 5), dtype=flow.float)):
            var = get_var("var", trainable=True)
            loss = var + x_def
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                momentum=0,
            ).minimize(loss)
            return var

        @flow.global_function()
        def eval():
            return get_var("var")

        variables = []
        for i in range(10):
            input = np.random.rand(2, 5).astype(np.single)
            eval_var = eval().get()
            train_var = train(input).get()
            # print("variable at iter {}:".format(i), var)
            test_case.assertTrue(np.array_equal(eval_var.numpy(), train_var.numpy()))
            if i > 0:
                test_case.assertTrue(
                    np.allclose(
                        eval_var.numpy(),
                        (variables[-1] - learning_rate / eval_var.size),
                    )
                )

            variables.append(eval_var.numpy())

    def test_get_job_inter_and_intra_shared_variable(test_case):
        flow.clear_default_session()

        variable_shape = (2, 5)

        def get_var(name, shape=variable_shape, dtype=flow.float, trainable=False):
            return flow.get_variable(
                name=name,
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                initializer=flow.random_uniform_initializer(),
            )

        learning_rate = 1e-2

        @flow.global_function(type="train", function_config=flow.FunctionConfig())
        def train(x_def: oft.Numpy.Placeholder(shape=variable_shape, dtype=flow.float)):
            var = get_var("var", trainable=True)
            loss = var + x_def
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [learning_rate]),
                momentum=0,
            ).minimize(loss)
            return var

        @flow.global_function()
        def eval():
            v1 = get_var("var")
            v2 = get_var("var")
            return v1, v2

        variables = []
        for i in range(10):
            input = np.random.rand(*variable_shape).astype(np.single)
            var1, var2 = eval().get()
            train_var = train(input).get()
            # print("variable at iter {}:".format(i), var1.numpy())
            test_case.assertTrue(np.array_equal(var1.numpy(), var2.numpy()))
            test_case.assertTrue(np.array_equal(var1.numpy(), train_var.numpy()))
            if i > 0:
                test_case.assertTrue(
                    np.allclose(
                        var1.numpy(), (variables[-1] - learning_rate / var1.size)
                    )
                )

            variables.append(var1.numpy())


if __name__ == "__main__":
    unittest.main()

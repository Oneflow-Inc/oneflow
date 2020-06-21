import oneflow as flow
import numpy as np


def test_naive_copy(test_case):
    flow.enable_eager_execution()

    @flow.global_function()
    def Foo():
        x = flow.get_variable(
            "x", shape=(10,), dtype=flow.float, initializer=flow.constant_initializer(0)
        )
        y = flow.copy(x)
        for x_numpy, y_numpy in zip(x.numpy_mirrored_list(), y.numpy_mirrored_list()):
            test_case.assertTrue(np.array_equal(x_numpy, y_numpy))

    Foo()


def test_copy_d2h(test_case):
    flow.enable_eager_execution()

    @flow.global_function()
    def Foo():
        with flow.fixed_placement("gpu", "0:0"):
            x = flow.get_variable(
                "x",
                shape=(10,),
                dtype=flow.float,
                initializer=flow.constant_initializer(0),
            )
        with flow.fixed_placement("cpu", "0:0"):
            y = flow.copy(x)
        for x_numpy, y_numpy in zip(x.numpy_mirrored_list(), y.numpy_mirrored_list()):
            test_case.assertTrue(np.array_equal(x_numpy, y_numpy))

    Foo()


def test_copy_h2d(test_case):
    flow.enable_eager_execution()

    @flow.global_function()
    def Foo():
        with flow.fixed_placement("cpu", "0:0"):
            x = flow.get_variable(
                "x",
                shape=(10,),
                dtype=flow.float,
                initializer=flow.constant_initializer(0),
            )
        with flow.fixed_placement("gpu", "0:0"):
            y = flow.copy(x)
        for x_numpy, y_numpy in zip(x.numpy_mirrored_list(), y.numpy_mirrored_list()):
            test_case.assertTrue(np.array_equal(x_numpy, y_numpy))

    Foo()

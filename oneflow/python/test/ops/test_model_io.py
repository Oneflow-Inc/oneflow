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
import oneflow.typing as tp
import time
import os


def _make_gen_var_func(shape, dtype, lr):
    @flow.global_function(type="train")
    def gen_var(x: tp.Numpy.Placeholder(shape=shape, dtype=dtype)) -> tp.Numpy:
        var = flow.get_variable(
            name="var",
            shape=shape,
            dtype=dtype,
            initializer=flow.random_uniform_initializer(),
        )
        y = var + x
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [lr]), momentum=0
        ).minimize(y)
        return var

    return gen_var


def _make_get_var_func(shape, dtype):
    @flow.global_function(type="predict")
    def get_var() -> tp.Numpy:
        return flow.get_variable(
            name="var",
            shape=shape,
            dtype=dtype,
            initializer=flow.random_uniform_initializer(),
            reuse=True,
        )

    return get_var


def _load_snapshot_manually(path, shape, dtype):
    var_path = os.path.join(path, "var", "out")
    return np.fromfile(
        var_path, dtype=flow.convert_oneflow_dtype_to_numpy_dtype(dtype)
    ).reshape(*shape)


def _test_model_io(test_case, shape, dtype, lr, num_iters):
    flow.clear_default_session()
    flow.config.enable_legacy_model_io(True)
    gen_var = _make_gen_var_func(shape, dtype, lr)

    model_save_root_dir = "./log/snapshot/"
    if not os.path.exists(model_save_root_dir):
        os.makedirs(model_save_root_dir)
    snapshot_path = model_save_root_dir + "snapshot-{}".format(
        time.strftime("%Y%m%d-%H:%M:%S")
    )
    checkpoint = flow.train.CheckPoint()
    checkpoint.init()

    variables = []
    for i in range(num_iters):
        var = gen_var(
            np.random.rand(*shape).astype(
                flow.convert_oneflow_dtype_to_numpy_dtype(dtype)
            )
        )
        if i > 0:
            test_case.assertTrue(np.allclose(var, (variables[-1] - lr / var.size)))

        variables.append(var)
        checkpoint.save("{}-{}".format(snapshot_path, i))

    flow.clear_default_session()
    get_var = _make_get_var_func(shape, dtype)

    final_snapshot_path = "{}-{}".format(snapshot_path, num_iters - 1)
    checkpoint = flow.train.CheckPoint()
    checkpoint.load(final_snapshot_path)

    final_var = get_var()
    var_from_file = _load_snapshot_manually(final_snapshot_path, shape, dtype)
    test_case.assertTrue(np.allclose(final_var, var_from_file))


@flow.unittest.skip_unless_1n1d()
class TestModelIo(flow.unittest.TestCase):
    def test_model_io_case_0(test_case):
        if flow.eager_execution_enabled():
            print("\nSkip under erger mode!")
            return
        # _test_model_io(test_case, (10, 5, 7), flow.float32, 1e-2, 10)
        _test_model_io(test_case, (2, 2), flow.float32, 1e-2, 10)


if __name__ == "__main__":
    unittest.main()

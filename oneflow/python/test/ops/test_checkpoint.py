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
import os
import shutil
import tempfile

import numpy as np
import oneflow as flow
import oneflow.typing as tp


def refresh_session():
    flow.clear_default_session()
    flow.config.gpu_device_num(flow.unittest.env.device_num())


def get_placement():
    node_size = flow.unittest.env.node_size()
    device_ids = "0-{}".format(flow.unittest.env.device_num() - 1)
    machine_device_ids = [
        "{}:{}".format(node_id, device_ids) for node_id in range(node_size)
    ]
    return flow.scope.placement("gpu", machine_device_ids)


def get_simple_momentum_training_model(dtype):
    assert dtype == flow.float32

    @flow.global_function(type="train")
    def model() -> tp.Numpy:
        with get_placement():
            x = flow.get_variable(
                name="x",
                shape=(4, 5),
                dtype=flow.float32,
                initializer=flow.random_normal_initializer(mean=10, stddev=1),
            )
            w = flow.get_variable(
                name="w",
                shape=(5, 6),
                dtype=flow.float32,
                initializer=flow.random_normal_initializer(mean=10, stddev=1),
                distribute=flow.distribute.split(0),
            )
            y = flow.matmul(x, w)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0.01]), momentum=0.9
            ).minimize(y)
            return y

    return model


def get_simple_model(dtype):
    @flow.global_function()
    def add() -> tp.Numpy:
        with get_placement():
            x = flow.get_variable(
                name="x",
                shape=(9, 3),
                dtype=dtype,
                initializer=flow.random_normal_initializer(mean=10, stddev=1),
                distribute=flow.distribute.split(0),
            )
            y = flow.get_variable(
                name="y",
                shape=(9, 3),
                dtype=dtype,
                initializer=flow.constant_initializer(5, dtype=dtype),
            )
            z = flow.get_variable(
                name="z",
                shape=(9, 3),
                dtype=dtype,
                initializer=flow.random_normal_initializer(),
            )
            return flow.math.add_n([x, y, z])

    return add


def get_large_model(dtype):
    @flow.global_function()
    def large() -> tp.Numpy:
        with get_placement():
            x = flow.get_variable(
                name="x",
                shape=(10, 2801, 820, 4),
                dtype=dtype,
                initializer=flow.random_normal_initializer(mean=10, stddev=1),
                distribute=flow.distribute.split(0),
            )
            return flow.math.reduce_mean(x)

    return large


def get_add_and_reduce_mean_model(dtype):
    @flow.global_function()
    def model() -> tp.Numpy:
        with get_placement():
            x = flow.get_variable(
                name="x",
                shape=(10, 801, 820, 4),
                dtype=dtype,
                initializer=flow.random_normal_initializer(mean=10, stddev=1),
                distribute=flow.distribute.split(0),
            )
            y = flow.get_variable(
                name="y",
                shape=(10, 801, 820, 4),
                dtype=dtype,
                initializer=flow.random_normal_initializer(mean=10, stddev=1),
                distribute=flow.distribute.split(0),
            )
            return flow.math.reduce_mean(x + y)

    return model


def get_checkpoint_ready_model(model_getter, dtype):
    model = model_getter(dtype)
    if flow.eager_execution_enabled():
        model()
    return model


def _TestSaveCorrectness(test_case, model_getter, dtype, legacy_api):
    """
    Save weights by new model io, load weights by legacy model io,
    and check the equality.
    """
    with tempfile.TemporaryDirectory() as save_dir:
        refresh_session()
        flow.config.enable_legacy_model_io(False)

        large1 = get_checkpoint_ready_model(model_getter, dtype)

        if legacy_api:
            check_point = flow.train.CheckPoint()
            check_point.save(save_dir)
        else:
            flow.checkpoint.save(save_dir)
        res1 = large1()

        refresh_session()
        flow.config.enable_legacy_model_io(True)

        large2 = get_checkpoint_ready_model(model_getter, dtype)

        check_point = flow.train.CheckPoint()
        check_point.load(save_dir)
        flow.sync_default_session()

        res2 = large2()
        test_case.assertTrue(np.array_equal(res1, res2))


def _TestRoundTrip(test_case, model_getter, dtype):
    """
    Save weights by new model io, load weights by new model io,
    and check the equality.
    """
    with tempfile.TemporaryDirectory() as save_dir:
        refresh_session()

        large1 = get_checkpoint_ready_model(model_getter, dtype)

        flow.checkpoint.save(save_dir)
        res1 = large1()

        refresh_session()

        large2 = get_checkpoint_ready_model(model_getter, dtype)

        vars_in_file = flow.checkpoint.get(save_dir)
        flow.load_variables(vars_in_file)
        res2 = large2()

        test_case.assertTrue(np.array_equal(res1, res2))


def _TestLoadCorrectness(test_case, model_getter, dtype, legacy_api):
    """
    Save weights by legacy model io, load weights by new model io,
    and check the equality.
    """
    with tempfile.TemporaryDirectory() as save_dir:
        refresh_session()
        flow.config.enable_legacy_model_io(True)

        large1 = get_checkpoint_ready_model(model_getter, dtype)

        check_point = flow.train.CheckPoint()
        check_point.init()

        check_point.save(save_dir)
        res1 = large1()

        flow.clear_default_session()
        flow.config.gpu_device_num(4)
        flow.config.enable_legacy_model_io(False)

        large2 = get_checkpoint_ready_model(model_getter, dtype)

        if legacy_api:
            check_point = flow.train.CheckPoint()
            check_point.load(save_dir)
        else:
            vars_in_file = flow.checkpoint.get(save_dir)
            flow.load_variables(vars_in_file)

        res2 = large2()

        test_case.assertTrue(np.array_equal(res1, res2))


def _TestPartiallyLoadNumpy(test_case, dtype):
    refresh_session()

    model = get_checkpoint_ready_model(get_add_and_reduce_mean_model, dtype)
    var_x = flow.get_all_variables()["x"]
    var_y_value_before_loading = flow.get_all_variables()["y"].numpy()
    new_val_np = np.random.random(var_x.shape).astype(np.float32)
    flow.load_variables({"x": new_val_np})
    var_y_value_after_loading = flow.get_all_variables()["y"].numpy()
    flow_res = model()
    np_res = (var_y_value_after_loading + new_val_np).mean()
    test_case.assertTrue(np.allclose(flow_res, np_res))
    test_case.assertTrue(
        np.array_equal(var_y_value_before_loading, var_y_value_after_loading)
    )


def _TestMixedModel(test_case, dtype):
    with tempfile.TemporaryDirectory() as save_dir1, tempfile.TemporaryDirectory() as save_dir2:

        def get_variable(name):
            return flow.get_variable(
                name=name,
                shape=(10, 80, 40, 20),
                dtype=dtype,
                initializer=flow.random_normal_initializer(mean=10, stddev=1),
                distribute=flow.distribute.split(0),
            )

        def get_part_of_mixed_model(dtype):
            @flow.global_function()
            def model() -> tp.Numpy:
                with get_placement():
                    x = get_variable("x")
                    return x

            return model

        def get_mixed_model(dtype):
            @flow.global_function()
            def model() -> tp.Numpy:
                with get_placement():
                    x1 = get_variable("x_from_model1")
                    x2 = get_variable("x_from_model2")
                    return x1 + x2

            return model

        refresh_session()
        model1 = get_checkpoint_ready_model(get_part_of_mixed_model, dtype)
        flow.checkpoint.save(save_dir1)

        refresh_session()
        model2 = get_checkpoint_ready_model(get_part_of_mixed_model, dtype)
        flow.checkpoint.save(save_dir2)

        refresh_session()
        mixed_model = get_checkpoint_ready_model(get_mixed_model, dtype)
        var_dict_from_model1 = flow.checkpoint.get(save_dir1)
        var_dict_from_model2 = flow.checkpoint.get(save_dir2)
        new_var_dict = {}
        for key, val in var_dict_from_model1.items():
            new_var_dict["{}_from_model1".format(key)] = val
        for key, val in var_dict_from_model2.items():
            new_var_dict["{}_from_model2".format(key)] = val
        flow.load_variables(new_var_dict)
        res = mixed_model()
        test_case.assertTrue(
            np.allclose(
                res,
                var_dict_from_model1["x"].numpy() + var_dict_from_model2["x"].numpy(),
            )
        )


def _TestResumeTraining(test_case):
    with tempfile.TemporaryDirectory() as save_dir:
        refresh_session()
        model = get_checkpoint_ready_model(
            get_simple_momentum_training_model, flow.float32
        )
        model()
        flow.checkpoint.save(save_dir)
        model()
        w1 = flow.get_all_variables()["w"].numpy()

        refresh_session()
        model = get_checkpoint_ready_model(
            get_simple_momentum_training_model, flow.float32
        )
        flow.load_variables(flow.checkpoint.get(save_dir))
        model()
        w2 = flow.get_all_variables()["w"].numpy()

        test_case.assertTrue(np.array_equal(w1, w2))


def _TestAssignmentBetweenMemory(test_case, dtype):
    refresh_session()

    model = get_checkpoint_ready_model(get_simple_model, dtype)
    all_vars = flow.get_all_variables()
    flow.load_variables({"x": all_vars["z"]})
    flow_res = model()
    np_res = all_vars["z"].numpy() * 2 + all_vars["y"].numpy()
    test_case.assertTrue(np.allclose(flow_res, np_res))


class TestCheckpoint(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    @unittest.skipIf(
        flow.unittest.env.eager_execution_enabled(),
        "legacy model io doesn't work in eager mode",
    )
    def test_save_correctness_1node_legacy_api(test_case):
        _TestSaveCorrectness(test_case, get_simple_model, flow.float, True)

    @flow.unittest.skip_unless_1n4d()
    @unittest.skipIf(
        flow.unittest.env.eager_execution_enabled(),
        "legacy model io doesn't work in eager mode",
    )
    def test_load_correctness_1node_legacy_api(test_case):
        _TestLoadCorrectness(test_case, get_simple_model, flow.float, True)

    @flow.unittest.skip_unless_1n4d()
    @unittest.skipIf(
        flow.unittest.env.eager_execution_enabled(),
        "legacy model io doesn't work in eager mode",
    )
    def test_save_correctness_1node(test_case):
        for dtype in [flow.float, flow.double]:
            _TestSaveCorrectness(test_case, get_large_model, dtype, False)

    @flow.unittest.skip_unless_2n4d()
    @unittest.skipIf(
        flow.unittest.env.eager_execution_enabled(),
        "legacy model io doesn't work in eager mode",
    )
    def test_save_correctness_2node(test_case):
        _TestSaveCorrectness(test_case, get_large_model, flow.float, False)

    @flow.unittest.skip_unless_1n4d()
    @unittest.skipIf(
        flow.unittest.env.eager_execution_enabled(),
        "legacy model io doesn't work in eager mode",
    )
    def test_load_correctness_1node(test_case):
        for dtype in [flow.float, flow.double]:
            _TestLoadCorrectness(test_case, get_large_model, dtype, False)

    @flow.unittest.skip_unless_2n4d()
    @unittest.skipIf(
        flow.unittest.env.eager_execution_enabled(),
        "legacy model io doesn't work in eager mode",
    )
    def test_load_correctness_2node(test_case):
        _TestLoadCorrectness(test_case, get_large_model, flow.float, False)

    @flow.unittest.skip_unless_1n4d()
    def test_assignment_between_memory(test_case):
        _TestAssignmentBetweenMemory(test_case, flow.float)

    @flow.unittest.skip_unless_1n4d()
    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "Save and load are covered by other tests in lazy mode",
    )
    def test_round_trip(test_case):
        _TestRoundTrip(test_case, get_large_model, flow.float)

    @flow.unittest.skip_unless_1n4d()
    def test_partially_load_numpy(test_case):
        _TestPartiallyLoadNumpy(test_case, flow.float)

    @flow.unittest.skip_unless_1n2d()
    def test_mixed_model(test_case):
        _TestMixedModel(test_case, flow.float)

    @flow.unittest.skip_unless_1n2d()
    def test_resume_training(test_case):
        _TestResumeTraining(test_case)


if __name__ == "__main__":
    unittest.main()

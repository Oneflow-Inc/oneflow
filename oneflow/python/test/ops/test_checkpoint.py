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


def get_placement():
    node_size = flow.unittest.env.node_size()
    device_ids = "0-{}".format(flow.unittest.env.device_num() - 1)
    machine_device_ids = [
        "{}:{}".format(node_id, device_ids) for node_id in range(node_size)
    ]
    return flow.scope.placement("gpu", machine_device_ids)


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
                shape=(8801, 8203, 4),
                dtype=dtype,
                initializer=flow.random_normal_initializer(mean=10, stddev=1),
                distribute=flow.distribute.split(0),
            )
            return flow.math.reduce_mean(x)

    return large


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
        flow.clear_default_session()
        flow.config.gpu_device_num(4)
        flow.config.enable_legacy_model_io(False)

        large1 = get_checkpoint_ready_model(model_getter, dtype)

        if legacy_api:
            check_point = flow.train.CheckPoint()
            check_point.save(save_dir)
        else:
            flow.save(flow.get_all_variables(), save_dir)
        res1 = large1()

        flow.clear_default_session()
        flow.config.gpu_device_num(4)
        flow.config.enable_legacy_model_io(True)

        large2 = get_checkpoint_ready_model(model_getter, dtype)

        check_point = flow.train.CheckPoint()
        check_point.load(save_dir)
        flow.sync_default_session()

        res2 = large2()
        test_case.assertTrue(np.array_equal(res1, res2))


def _TestLoadCorrectness(test_case, model_getter, dtype, legacy_api):
    """
    Save weights by legacy model io, load weights by new model io,
    and check the equality.
    """
    with tempfile.TemporaryDirectory() as save_dir:
        flow.clear_default_session()
        flow.config.gpu_device_num(4)
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
            vars_in_file = flow.load(save_dir)
            flow.load_variables(vars_in_file)

        res2 = large2()

        test_case.assertTrue(np.array_equal(res1, res2))


class TestCheckpoint(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n4d()
    def test_save_correctness_1node_legacy_api(test_case):
        _TestSaveCorrectness(test_case, get_simple_model, flow.float, True)

    @flow.unittest.skip_unless_1n4d()
    def test_load_correctness_1node_legacy_api(test_case):
        _TestLoadCorrectness(test_case, get_simple_model, flow.float, True)

    @flow.unittest.skip_unless_1n4d()
    def test_save_correctness_1node(test_case):
        for dtype in [flow.float, flow.double]:
            _TestSaveCorrectness(test_case, get_large_model, dtype, False)

    @flow.unittest.skip_unless_2n4d()
    def test_save_correctness_2node(test_case):
        _TestSaveCorrectness(test_case, get_large_model, flow.float, False)

    @flow.unittest.skip_unless_1n4d()
    def test_load_correctness_1node(test_case):
        for dtype in [flow.float, flow.double]:
            _TestLoadCorrectness(test_case, get_large_model, dtype, False)

    @flow.unittest.skip_unless_2n4d()
    def test_load_correctness_2node(test_case):
        _TestLoadCorrectness(test_case, get_large_model, flow.float, False)


if __name__ == "__main__":
    unittest.main()

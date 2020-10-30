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


def get_simple_model():
    @flow.global_function()
    def add() -> tp.Numpy:
        with get_placement():
            x = flow.get_variable(
                name="x",
                shape=(4, 3),
                initializer=flow.random_normal_initializer(mean=10, stddev=1),
                distribute=flow.distribute.split(0),
                # distribute=flow.distribute.broadcast(),
            )
            y = flow.get_variable(
                name="y", shape=(4, 3), initializer=flow.constant_initializer(5),
            )
            z = flow.get_variable(
                name="z", shape=(4, 3), initializer=flow.random_normal_initializer(),
            )
            return flow.math.add_n([x, y, z])

    return add


def get_large_model():
    @flow.global_function()
    def large() -> tp.Numpy:
        with get_placement():
            x = flow.get_variable(
                name='x',
                shape=(28901, 8203, 4),
                initializer=flow.random_normal_initializer(mean=10, stddev=1),
                distribute=flow.distribute.split(0),
            )
            return flow.math.reduce_mean(x)

    return large


def get_checkpoint_ready_model(model_getter):
    model = model_getter()
    if flow.eager_execution_enabled():
        model()
    return model


def _TestLegacyAPI(test_case, legacy_model_io_enabled):
    flow.clear_default_session()
    flow.config.gpu_device_num(2)
    flow.config.enable_legacy_model_io(legacy_model_io_enabled)

    add1 = get_checkpoint_ready_model(get_simple_model)

    check_point = flow.train.CheckPoint()
    check_point.init()
    save_dir = "/tmp/legacy_cp"
    shutil.rmtree(save_dir, ignore_errors=True)
    check_point.save(save_dir)
    flow.sync_default_session()
    res1 = add1()

    flow.clear_default_session()
    flow.config.gpu_device_num(2)

    add2 = get_checkpoint_ready_model(get_simple_model)

    check_point.load(save_dir)
    flow.sync_default_session()
    res2 = add2()

    test_case.assertTrue(np.array_equal(res1, res2))


def _TestCorrectness(test_case, model_getter):
    flow.clear_default_session()
    flow.config.gpu_device_num(4)
    flow.config.enable_legacy_model_io(False)

    large1 = get_checkpoint_ready_model(model_getter)

    if flow.config.legacy_model_io_enabled():
        check_point = flow.train.CheckPoint()
        check_point.init()
    check_point = flow.train.CheckPoint()
    check_point.init()

    vars_in_mem = flow.get_all_variables()
    save_dir = "/tmp/cp"
    shutil.rmtree(save_dir, ignore_errors=True)
    flow.save(vars_in_mem, save_dir)
    res1 = large1()

    flow.clear_default_session()
    flow.config.gpu_device_num(4)

    large2 = get_checkpoint_ready_model(model_getter)

    vars_in_file = flow.load(save_dir)
    flow.load_variables(vars_in_file)

    res2 = large2()
    print(res1)
    print(res2)
    test_case.assertTrue(np.array_equal(res1, res2))


def _Test(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(2)

    add = get_checkpoint_ready_model(get_simple_model)

    check_point = flow.train.CheckPoint()
    if flow.config.legacy_model_io_enabled():
        check_point.init()

    vars_in_mem = flow.get_all_variables()
    print(vars_in_mem)
    flow.load_variables({"y": vars_in_mem["x"]})
    test_case.assertTrue(
        np.array_equal(vars_in_mem["y"].numpy(), vars_in_mem["x"].numpy())
    )

    if flow.config.legacy_model_io_enabled():
        save_dir = "/tmp/legacy_cp"
        shutil.rmtree(save_dir, ignore_errors=True)
        check_point.save(save_dir)
        flow.sync_default_session()
    else:
        save_dir = "/tmp/cp"
        shutil.rmtree(save_dir, ignore_errors=True)
        flow.save(vars_in_mem, save_dir)

    vars_in_file = flow.load(save_dir)
    test_case.assertTrue(
        np.array_equal(vars_in_mem["x"].numpy(), vars_in_file["x"].numpy())
    )
    test_case.assertTrue(
        np.array_equal(vars_in_mem["y"].numpy(), vars_in_file["y"].numpy())
    )
    test_case.assertTrue(
        np.array_equal(vars_in_mem["z"].numpy(), vars_in_file["z"].numpy())
    )
    flow.load_variables({"y": vars_in_file["z"]})
    test_case.assertTrue(
        np.array_equal(vars_in_mem["y"].numpy(), vars_in_file["z"].numpy())
    )

    net_result = add()
    np_result = (
        vars_in_mem["x"].numpy() + vars_in_mem["y"].numpy() + vars_in_mem["z"].numpy()
    )
    test_case.assertTrue(np.array_equal(net_result, np_result))


class TestCheckpoint(flow.unittest.TestCase):
    @flow.unittest.skip_unless_2n2d()
    def test_2nodes(test_case):
        _Test(test_case)

    @flow.unittest.skip_unless_1n2d()
    def test_1node(test_case):
        _Test(test_case)

    @flow.unittest.skip_unless_1n2d()
    def test_legacy_api_1node(test_case):
        _TestLegacyAPI(test_case, False)

    @flow.unittest.skip_unless_1n4d()
    def test_correctness_1node(test_case):
        _TestCorrectness(test_case, get_large_model)

    @flow.unittest.skip_unless_2n4d()
    def test_correctness_2node(test_case):
        _TestCorrectness(test_case, get_large_model)


if __name__ == "__main__":
    unittest.main()

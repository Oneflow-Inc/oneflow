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
    if node_size == 1:
        return flow.scope.placement("gpu", "0:0-1")
    elif node_size == 2:
        return flow.scope.placement("gpu", ["0:0-1", "1:0-1"])
    else:
        raise RuntimeError("Invalid node size {}".format(node_size))


def _Test(test_case):
    flow.clear_default_session()
    flow.config.gpu_device_num(2)
    flow.use_legacy_checkpoint(False)

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
                name="z",
                shape=(4, 3),
                initializer=flow.random_normal_initializer(),
            )
            return flow.math.add_n([x, y, z])

    if flow.eager_execution_enabled():
        add()

    check_point = flow.train.CheckPoint()
    if flow.legacy_checkpoint_used():
        check_point.init()

    vars_in_mem = flow.get_all_variables()
    print(vars_in_mem)
    flow.checkpoint.load_variables({"y": vars_in_mem["x"]})
    test_case.assertTrue(
        np.array_equal(vars_in_mem["y"].numpy(), vars_in_mem["x"].numpy())
    )

    if flow.legacy_checkpoint_used():
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
    flow.checkpoint.load_variables({"y": vars_in_file["z"]})
    test_case.assertTrue(
        np.array_equal(vars_in_mem["y"].numpy(), vars_in_file["z"].numpy())
    )

    net_result = add()
    np_result = (
        vars_in_mem["x"].numpy()
        + vars_in_mem["y"].numpy()
        + vars_in_mem["z"].numpy()
    )
    test_case.assertTrue(np.array_equal(net_result, np_result))


class TestCheckpoint(flow.unittest.TestCase):
    @flow.unittest.skip_unless_2n2d()
    def test_2nodes(test_case):
        _Test(test_case)


    @flow.unittest.skip_unless_1n2d()
    def test_1node(test_case):
        _Test(test_case)

if __name__ == "__main__":
    unittest.main()

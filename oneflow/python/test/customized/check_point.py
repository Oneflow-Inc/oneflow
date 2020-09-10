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
import os
import shutil

import numpy as np

import oneflow as flow
import oneflow.typing as tp


flow.config.gpu_device_num(2)
flow.enable_eager_execution(False)

@flow.global_function()
def add() -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0-1"):
        x = flow.get_variable(
            name="x", shape=(2, 3), initializer=flow.random_normal_initializer(),
        )
        y = flow.get_variable(
            name="y", shape=(2, 3), initializer=flow.random_uniform_initializer(),
        )
        z = flow.get_variable(
            name="z", shape=(2, 3), initializer=flow.xavier_uniform_initializer(),
        )
        return flow.math.add_n([x, y, z])
        return z


if flow.eager_execution_enabled():
    add()
check_point = flow.train.CheckPoint()
check_point.init()

all_vars = flow.get_all_variables()
print(all_vars)
print("--------")

legacy = False
if legacy:
    save_dir = "/tmp/legacy_cp"
    shutil.rmtree(save_dir)
    check_point.save(save_dir)
    flow.sync_default_session()
else:
    save_dir = "/tmp/cp"
    flow.save(all_vars, save_dir)

vars = flow.load(save_dir)
print(vars)
print("--------")
flow.checkpoint.load_variables({"y": vars["x"]})

print(flow.get_all_variables())
print(add())

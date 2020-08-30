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


@flow.global_function()
def add() -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0-1"):
        x = flow.get_variable(
            name="x", shape=(2, 3), initializer=flow.random_uniform_initializer(),
        )
        y = flow.get_variable(
            name="y", shape=(2, 3), initializer=flow.random_uniform_initializer(),
        )
        z = flow.get_variable(
            name="z", shape=(2, 3), initializer=flow.random_uniform_initializer(),
        )
        return flow.math.add_n([x, y, z])
        return z


check_point = flow.train.CheckPoint()
check_point.init()

print(flow.get_all_variables())

save_dir = "/tmp/legacy_cp"

# flow.checkpoint.save_all_variables(save_dir)
check_point.save(save_dir)
flow.sync_default_session()

vars = flow.load(save_dir)
print(vars)
flow.checkpoint.load_variables({'y': vars['x']})

print(flow.get_all_variables())
print(add())

shutil.rmtree(save_dir)

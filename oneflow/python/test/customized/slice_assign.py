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
import oneflow as flow
import numpy as np
import oneflow.typing as oft

var_shape = (30, 40, 20, 15)
value = np.random.rand(9, 8, 7, 6).astype(np.float32)


@flow.global_function()
def assign_fn(value_def: oft.Numpy.Placeholder(value.shape)):
    with flow.scope.placement("cpu", "0:0-3"):
        var = flow.get_variable(
            name="var",
            shape=var_shape,
            dtype=flow.float32,
            initializer=flow.constant_initializer(0),
            distribute=flow.distribute.split(1),
        )
        flow.slice_assign(
            var, value_def, [(10, 19, 1), (1, 30, 4), (3, 16, 2), (5, 11, 1)]
        )
        return var


@flow.global_function()
def identity_fn():
    with flow.scope.placement("cpu", "0:0-3"):
        var = flow.get_variable(
            name="var",
            shape=var_shape,
            dtype=flow.float32,
            initializer=flow.constant_initializer(0),
            distribute=flow.distribute.split(1),
        )
        return flow.identity(var)


assign_fn(value)
of_res = identity_fn().get().numpy()

np_res = np.zeros(var_shape)
np_res[10:19, 1:30:4, 3:16:2, 5:11] = value

print(np.array_equal(of_res, np_res))

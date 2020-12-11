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
import oneflow.typing as tp
import numpy as np

global_storage = {}

def Get(name):
    return global_storage.get(name).numpy()

def Setter(name):
    global global_storage
    def _set(x):
        global_storage[name] = x
    return _set

# @flow.global_function()
# def pad_Job(x: tp.Numpy.Placeholder((1, 2, 3, 3))) -> tp.Numpy:
#     with flow.scope.placement("cpu", "0:0"):
#         loss = flow.reflection_pad2d(x, padding=[0, 0, 1, 2], data_format="NCHW")
#         return loss

# x = np.arange(18).reshape((1, 2, 3, 3)).astype(np.float32)
# out = pad_Job(x)
# print("in:\n", x)
# print("out:\n", out)


flow.clear_default_session()
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

@flow.global_function(type="train", function_config=func_config)
def pad_grad_Job(x: tp.Numpy.Placeholder((1, 1, 3, 3))) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        x += flow.get_variable(
            name="v1",
            shape=(1,),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        loss = flow.reflection_pad2d(x, padding=[0, 0, 2, 2], data_format="NCHW")
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [0]), momentum=0
        ).minimize(loss)
        flow.watch_diff(x, Setter("x_diff"))
        return loss

# OneFlow
check_point = flow.train.CheckPoint()
check_point.init()

x = np.arange(9).reshape((1, 1, 3, 3)).astype(np.float32)
out = pad_grad_Job(x)
out_grad = Get("x_diff")
print("in:\n", x)
print("out:\n", out, "\n out_grad:\n", out_grad)
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

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


@flow.global_function(function_config=func_config)
def jobA(input: tp.Numpy.Placeholder((1, 10), dtype=flow.float32)) -> tp.Numpy:
    with flow.scope.consistent_view():
        print(input.logical_blob_name)
        var = flow.get_variable(
            "job_a_var",
            shape=input.shape,
            dtype=input.dtype,
            initializer=flow.ones_initializer(),
            trainable=False,
        )
        output = input + var
        print(output.logical_blob_name)
        return output


@flow.global_function(function_config=func_config)
def jobB(input: tp.Numpy.Placeholder((1, 10), dtype=flow.float32)) -> tp.Numpy:
    with flow.scope.consistent_view():
        print(input.logical_blob_name)
        var = flow.get_variable(
            "job_b_var",
            shape=input.shape,
            dtype=input.dtype,
            initializer=flow.ones_initializer(),
            trainable=False,
        )
        output = input * var
        print(output.logical_blob_name)
        return output


@flow.global_function(function_config=func_config)
def jobC(input: tp.Numpy.Placeholder((1, 10), dtype=flow.float32)) -> tp.Numpy:
    with flow.scope.consistent_view():
        print(input.logical_blob_name)
        var = flow.get_variable(
            "job_c_var",
            shape=input.shape,
            dtype=input.dtype,
            initializer=flow.ones_initializer(),
            trainable=False,
        )
        output = input - var
        print(output.logical_blob_name)
        return output


@flow.global_function(function_config=func_config)
def jobD(input: tp.Numpy.Placeholder((1, 10), dtype=flow.float32)) -> tp.Numpy:
    with flow.scope.consistent_view():
        print(input.logical_blob_name)
        var = flow.get_variable(
            "job_d_var",
            shape=input.shape,
            dtype=input.dtype,
            initializer=flow.ones_initializer(),
            trainable=False,
        )
        output = input / var
        print(output.logical_blob_name)
        return output


@flow.global_function(function_config=func_config)
def jobE(
    input1: tp.Numpy.Placeholder((1, 10), dtype=flow.float32),
    input2: tp.Numpy.Placeholder((1, 10), dtype=flow.float32),
) -> tp.Numpy:
    with flow.scope.consistent_view():
        print(input1.logical_blob_name)
        print(input2.logical_blob_name)
        var = flow.get_variable(
            "job_e_var",
            shape=input1.shape,
            dtype=input1.dtype,
            initializer=flow.ones_initializer(),
            trainable=False,
        )
        output = input1 + var - input2
        print(output.logical_blob_name)
        return output


check_point = flow.train.CheckPoint()
check_point.init()

saved_model_builder = flow.saved_model.SavedModelBuilder("./saved_job_graph", 1)
saved_model_builder.ModelName("save_job_graphs_example").AddJobFunction(
    jobA, ["Input_0/out"], ["broadcast_add_1/z_0"]
).AddJobFunction(jobB, ["Input_3/out"], ["broadcast_mul_4/z_0"]).AddJobFunction(
    jobC, ["Input_6/out"], ["broadcast_sub_7/z_0"]
).AddJobFunction(
    jobD, ["Input_9/out"], ["broadcast_div_10/z_0"]
).AddJobFunction(
    jobE, ["Input_12/out", "Input_13/out"], ["broadcast_sub_15/z_0"]
).AddJobGraph(
    "Job_Graph_1",
    [jobA],
    [jobE],
    [
        ((jobA, None), ("Input_0/out", None)),
        ((jobB, jobA), ("Input_3/out", "broadcast_add_1/z_0")),
        ((jobC, jobB), ("Input_6/out", "broadcast_mul_4/z_0")),
        ((jobD, jobA), ("Input_9/out", "broadcast_add_1/z_0")),
        ((jobE, jobC), ("Input_12/out", "broadcast_sub_7/z_0")),
        ((jobE, jobD), ("Input_13/out", "broadcast_div_10/z_0")),
    ],
).Save()

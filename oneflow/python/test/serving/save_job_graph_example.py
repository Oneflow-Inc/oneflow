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
    jobA, {"input_A": "Input_0/out"}, {"output_A": "broadcast_add_1/z_0"}
).AddJobFunction(
    jobB, {"input_B": "Input_3/out"}, {"output_B": "broadcast_mul_4/z_0"}
).AddJobFunction(
    jobC, {"input_C": "Input_6/out"}, {"output_C": "broadcast_sub_7/z_0"}
).AddJobFunction(
    jobD, {"input_D": "Input_9/out"}, {"output_D": "broadcast_div_10/z_0"}
).AddJobFunction(
    jobE,
    {"input_E_1": "Input_12/out", "input_E_2": "Input_13/out"},
    {"output_E": "broadcast_sub_15/z_0"},
).AddJobGraph(
    "Job_Graph_1",
    [jobA],
    [jobE],
    [
        ((jobA, None), ("input_A", None)),
        ((jobB, jobA), ("input_B", "output_A")),
        ((jobC, jobB), ("input_C", "output_B")),
        ((jobD, jobA), ("input_D", "output_A")),
        ((jobE, jobC), ("input_E_1", "output_C")),
        ((jobE, jobD), ("input_E_2", "output_D")),
    ],
).Save()

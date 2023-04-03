// RUN: oneflow-opt %s \
// RUN: -aggregate-compute-ops \
// RUN: | FileCheck %s

// CHECK: %[[ARG0:[a-zA-Z0-9_]+]] = "oneflow.arg_sort"
// CHECK: %[[ARG1:[a-zA-Z0-9_]+]] = "oneflow.dim_gather"
// CHECK: "oneflow.output"(%[[ARG0]])

module {
  oneflow.job @GraphToRun_1(%arg0: tensor<2xf32>) -> (tensor<2xsi32>, tensor<2xf32>) {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_input.0.0_2", output_lbns = ["_GraphToRun_1_input.0.0_2/out"], scope_symbol_id = 30 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    %0 = "oneflow.relu"(%output) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %2 = "oneflow.arg_sort"(%1) {device_name = ["@0:0"], device_tag = "cuda", direction = "ASCENDING", hierarchy = [1], op_name = "arg_sort-2", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xsi32>
    %output_0 = "oneflow.output"(%2) {data_type = 5 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_output.0.0.1_4", output_lbns = ["_GraphToRun_1_output.0.0.1_4/out"], scope_symbol_id = 30 : i64, shape = [2 : si64]} : (tensor<2xsi32>) -> tensor<2xsi32>
    %3 = "oneflow.dim_gather"(%1, %2) {device_name = ["@0:0"], device_tag = "cuda", dim = 0 : si32, hierarchy = [1], op_name = "dim_gather-3", scope_symbol_id = 30 : i64} : (tensor<2xf32>, tensor<2xsi32>) -> tensor<2xf32>
    %output_1 = "oneflow.output"(%3) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_output.0.0.0_3", output_lbns = ["_GraphToRun_1_output.0.0.0_3/out"], scope_symbol_id = 30 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    oneflow.return %output_0, %output_1 : tensor<2xsi32>, tensor<2xf32>
  }
}

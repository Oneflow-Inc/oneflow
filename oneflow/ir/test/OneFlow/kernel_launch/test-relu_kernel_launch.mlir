// RUN: oneflow-opt %s \
// RUN: -wrap-ops-to-kernel-launch  -canonicalize | FileCheck %s

// CHECK: oneflow.kernel_launch

module {
  oneflow.job @GraphToRun_0(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    %output = "oneflow.input"(%arg0) {
      data_type = 2 : i32,
      device_name = ["@0:0"],
      device_tag = "cpu",
      hierarchy = [1],
      is_dynamic = false,
      nd_sbp = ["B"],
      op_name = "_GraphToRun_0_input.0.0_2",
      output_lbns = ["_GraphToRun_0_input.0.0_2/out"],
      scope_symbol_id = 12 : i64,
      shape = [1 : si64]
      } : (tensor<1xf32>) -> tensor<1xf32>
    %0 = "oneflow.relu"(%output) {
      device_name = ["@0:0"],
      device_tag = "cpu",
      hierarchy = [1],
      op_name = "relu-0",
      scope_symbol_id = 12 : i64
    } : (tensor<1xf32>) -> tensor<1xf32>
    %1 = "oneflow.relu"(%0) {
      device_name = ["@0:0"],
      device_tag = "cpu",
      hierarchy = [1],
      op_name = "relu-0",
      scope_symbol_id = 12 : i64
    } : (tensor<1xf32>) -> tensor<1xf32>
    %output_0 = "oneflow.output"(%1) {data_type = 2 : i32,
      device_name = ["@0:0"],
      device_tag = "cpu",
      hierarchy = [1],
      is_dynamic = false,
      nd_sbp = ["B"],
      op_name = "_GraphToRun_0_output.0.0_2",
      output_lbns = ["_GraphToRun_0_output.0.0_2/out"],
      scope_symbol_id = 12 : i64,
      shape = [1 : si64]
    } : (tensor<1xf32>) -> tensor<1xf32>
    oneflow.return %output_0 : tensor<1xf32>
  }
}

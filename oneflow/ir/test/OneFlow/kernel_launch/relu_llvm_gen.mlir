// RUN: oneflow-opt %s \
// RUN: -kernel-launch-with-llvm | FileCheck %s

// CHECK: oneflow.kernel_launch

module {
  func.func public @relu2D0(%arg0: tensor<1xf32>) -> tensor<1xf32> attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu2D0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
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
    %0 = oneflow.kernel_launch @relu2D0(%output) {
      device_name = ["@0:0"],
      device_tag = "cpu",
      hierarchy = [1],
      mlir_assembly = "\22func.func\22() ({\0A^bb0(%arg0: tensor<1xf32>):\0A  %0 = \22oneflow.relu\22(%arg0) {device_name = [\22@0:0\22], device_tag = \22cpu\22, hierarchy = [1], op_name = \22relu2D0\22, scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>\0A  \22func.return\22(%0) : (tensor<1xf32>) -> ()\0A}) {function_type = (tensor<1xf32>) -> tensor<1xf32>, llvm.emit_c_interface, sym_name = \22relu2D0\22, sym_visibility = \22public\22} : () -> ()",
      op_name = "relu2D0",
      scope_symbol_id = 12 : i64
    } : (tensor<1xf32>) -> tensor<1xf32>
    %output_0 = "oneflow.output"(%0) {
      data_type = 2 : i32,
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

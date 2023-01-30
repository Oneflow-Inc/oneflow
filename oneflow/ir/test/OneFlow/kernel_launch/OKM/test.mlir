// RUN: oneflow-opt %s \
// RUN: -extract-okm-tensor \
// RUN: -trim-return-to-void \
// RUN: -lower-to-okl \
// RUN: | FileCheck %s

module {
  func.func @_mlir_oneflow_subgraph0(%arg0: tensor<2xf32>) -> tensor<2xf32> attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
  oneflow.job @GraphToRun_0(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_input.0.0_2", output_lbns = ["_GraphToRun_0_input.0.0_2/out"], scope_symbol_id = 12 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    %0 = oneflow.kernel_launch @_mlir_oneflow_subgraph0(%output) {device_name = ["@0:0"], device_tag = "cpu", mlir_assembly = "", op_name = "_mlir_oneflow_subgraph0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %output_0 = "oneflow.output"(%0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_output.0.0_2", output_lbns = ["_GraphToRun_0_output.0.0_2/out"], scope_symbol_id = 12 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    oneflow.return %output_0 : tensor<2xf32>
  }
}

// RUN: oneflow-opt -lower-oneflow-to-tosa %s | FileCheck %s

module {
// CHECK:func @GraphModule_0(%arg0: tensor<1xf32>) -> tensor<1xf32>
  oneflow.job @GraphModule_0(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphModule_0_input.0.0_2", output_lbns = ["_GraphModule_0_input.0.0_2/out"], scope_symbol_id = 4611686018427412479 : i64, shape = [1 : si64]} : (tensor<1xf32>) -> tensor<1xf32>
    %0 = "oneflow.relu"(%output) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "fw.relu-relu-0", output_lbns = ["fw.relu-relu-0/y_0"], scope_symbol_id = 4611686018427424767 : i64} : (tensor<1xf32>) -> tensor<1xf32>
    %output_0 = "oneflow.output"(%0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphModule_0_output.0.0_2", output_lbns = ["_GraphModule_0_output.0.0_2/out"], scope_symbol_id = 4611686018427412479 : i64, shape = [1 : si64]} : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:return %0 : tensor<1xf32>
    oneflow.return %output_0 : tensor<1xf32>
  }
}

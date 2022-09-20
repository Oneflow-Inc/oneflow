// RUN: oneflow-opt %s \
// RUN: -convert-ofkl-callee-to-llvm | FileCheck %s

// CHECK: llvm.call @kernel_launch

module {
  func.func public @relu2D0(%arg0: tensor<1xf32>) -> tensor<1xf32> attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu2D0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

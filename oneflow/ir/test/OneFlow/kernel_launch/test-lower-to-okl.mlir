// RUN: oneflow-opt %s \
// RUN: -lower-to-okl  -canonicalize | FileCheck %s

// CHECK: oneflow.kernel_launch
module {
  func.func @wrap0(%arg0: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        op_name = "relu-0",
        scope_symbol_id = 12 : i64
    } : (tensor<1xf32>) -> tensor<1xf32>
    return %0, %0 : tensor<1xf32>, tensor<1xf32>
  }
}

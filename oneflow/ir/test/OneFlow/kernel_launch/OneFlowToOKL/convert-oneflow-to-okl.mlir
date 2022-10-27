// RUN: oneflow-opt %s \
// RUN: -extract-kernel-launch-tensor \
// RUN: -trim-return-as-void \
// RUN: -lower-to-okl \
// RUN: -split-into-funcs \
// RUN: -fetch-from-launcher \
// RUN: -only-keep-compute-ops \
// RUN: -mlir-print-ir-after-all \
// RUN: | FileCheck %s


// CHECK: module {
// CHECK:   func.func @okl_compute(%arg0: !okl.launcher_ctx) {
// CHECK:     %0 = "okl.fetch_run_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %1 = "okl.fetch_run_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %2 = "okl.fetch_kernel"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     %3 = "okl.fetch_kernel"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     "okl.launch"(%0, %2) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     "okl.launch"(%1, %3) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }


module {
  func.func @wrap0(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
}

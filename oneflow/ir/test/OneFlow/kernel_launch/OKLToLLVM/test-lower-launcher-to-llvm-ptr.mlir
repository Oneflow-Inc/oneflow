// RUN: oneflow-opt %s \
// RUN: -lower-launcher-to-llvm-ptr \
// RUN: | FileCheck %s

// CHECK: module {
// CHECK:   func.func @okl_compute(%[[ARG:[a-zA-Z0-9_]+]]: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
// CHECK:     %[[ARG0:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ARG]] : !llvm.ptr<i8> to !okl.launcher_ctx
// CHECK:     %[[ARG1:[a-zA-Z0-9_]+]] = "okl.fetch_run_ctx"(%[[ARG0]]) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %[[ARG2:[a-zA-Z0-9_]+]] = "okl.fetch_run_ctx"(%[[ARG0]]) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %[[ARG3:[a-zA-Z0-9_]+]] = "okl.fetch_kernel"(%[[ARG0]]) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     %[[ARG4:[a-zA-Z0-9_]+]] = "okl.fetch_kernel"(%[[ARG0]]) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     "okl.launch"(%[[ARG1]], %[[ARG3]]) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     "okl.launch"(%[[ARG2]], %[[ARG4]]) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }

module {
   func.func @okl_compute(%arg0: !okl.launcher_ctx) {
    %0 = "okl.fetch_run_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %1 = "okl.fetch_run_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %2 = "okl.fetch_kernel"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    %3 = "okl.fetch_kernel"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    "okl.launch"(%0, %2) : (!okl.run_ctx, !okl.kernel) -> ()
    "okl.launch"(%1, %3) : (!okl.run_ctx, !okl.kernel) -> ()
    return
  }
}

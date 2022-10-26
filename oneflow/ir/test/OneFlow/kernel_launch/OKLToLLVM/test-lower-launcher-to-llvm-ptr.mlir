// RUN: oneflow-opt %s \
// RUN: -lower-launcher-to-llvm-ptr \
// RUN: | FileCheck %s

// CHECK: module {
// CHECK:   func.func @okl_compute(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
// CHECK:     %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
// CHECK:     %1 = "okl.fetch_run_ctx"(%0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %2 = "okl.fetch_run_ctx"(%0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
// CHECK:     %3 = "okl.fetch_kernel"(%0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     %4 = "okl.fetch_kernel"(%0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
// CHECK:     "okl.launch"(%1, %3) : (!okl.run_ctx, !okl.kernel) -> ()
// CHECK:     "okl.launch"(%2, %4) : (!okl.run_ctx, !okl.kernel) -> ()
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

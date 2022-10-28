// RUN: oneflow-opt %s \
// RUN: -lower-okl-to-llvm-call \
// RUN: | FileCheck %s

// CHECK: module {
// CHECK:   llvm.func @launch(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
// CHECK:   llvm.func @fetch_kernel(!llvm.ptr<i8>, i64) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
// CHECK:   llvm.func @fetch_run_ctx(!llvm.ptr<i8>, i64) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
// CHECK:   func.func @okl_compute(%[[ARG:[a-zA-Z0-9_]+]]: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
// CHECK:     %[[ARG0:[a-zA-Z0-9_]+]] = builtin.unrealized_conversion_cast %[[ARG]] : !llvm.ptr<i8> to !okl.launcher_ctx
// CHECK:     %[[ARG1:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:     %[[ARG2:[a-zA-Z0-9_]+]] = llvm.call @fetch_run_ctx(%[[ARG]], %[[ARG1]]) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     %[[ARG3:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:     %[[ARG4:[a-zA-Z0-9_]+]] = llvm.call @fetch_run_ctx(%[[ARG]], %[[ARG3]]) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     %[[ARG5:[a-zA-Z0-9_]+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:     %[[ARG6:[a-zA-Z0-9_]+]] = llvm.call @fetch_kernel(%[[ARG]], %[[ARG5]]) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     %[[ARG7:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:     %[[ARG8:[a-zA-Z0-9_]+]] = llvm.call @fetch_kernel(%[[ARG]], %[[ARG7]]) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     llvm.call @launch(%[[ARG2]], %[[ARG6]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
// CHECK:     llvm.call @launch(%[[ARG4]], %[[ARG8]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
module {
  func.func @okl_compute(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
    %1 = "okl.fetch_run_ctx"(%0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %2 = "okl.fetch_run_ctx"(%0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %3 = "okl.fetch_kernel"(%0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    %4 = "okl.fetch_kernel"(%0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    "okl.launch"(%1, %3) : (!okl.run_ctx, !okl.kernel) -> ()
    "okl.launch"(%2, %4) : (!okl.run_ctx, !okl.kernel) -> ()
    return
  }
}

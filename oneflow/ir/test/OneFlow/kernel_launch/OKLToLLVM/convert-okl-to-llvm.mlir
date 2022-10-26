// RUN: oneflow-opt %s \
// RUN: -lower-launcher-to-llvm-ptr \
// RUN: -lower-okl-to-llvm-call \
// RUN: -reconcile-unrealized-casts \
// RUN: -convert-func-to-llvm \
// RUN: -mlir-print-ir-after-all \
// RUN: | FileCheck %s

// CHECK: module attributes {llvm.data_layout = ""} {
// CHECK:   llvm.func @launch(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
// CHECK:   llvm.func @fetch_kernel(!llvm.ptr<i8>, i64) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
// CHECK:   llvm.func @fetch_run_ctx(!llvm.ptr<i8>, i64) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
// CHECK:   llvm.func @okl_compute(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
// CHECK:     %0 = llvm.mlir.constant(0 : index) : i64
// CHECK:     %1 = llvm.call @fetch_run_ctx(%arg0, %0) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     %2 = llvm.mlir.constant(1 : index) : i64
// CHECK:     %3 = llvm.call @fetch_run_ctx(%arg0, %2) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     %4 = llvm.mlir.constant(0 : index) : i64
// CHECK:     %5 = llvm.call @fetch_kernel(%arg0, %4) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     %6 = llvm.mlir.constant(1 : index) : i64
// CHECK:     %7 = llvm.call @fetch_kernel(%arg0, %6) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     llvm.call @launch(%1, %5) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
// CHECK:     llvm.call @launch(%3, %7) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
// CHECK:     llvm.return
// CHECK:   }
// CHECK:   llvm.func @_mlir_ciface_okl_compute(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
// CHECK:     llvm.call @okl_compute(%arg0) : (!llvm.ptr<i8>) -> ()
// CHECK:     llvm.return
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

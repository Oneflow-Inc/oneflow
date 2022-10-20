// RUN: oneflow-opt %s \
// RUN: -lower-okl-to-llvm-call \
// RUN: | FileCheck %s

// CHECK: module {
// CHECK:   llvm.func @launch(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
// CHECK:   llvm.func @fetch_kernel(!llvm.ptr<i8>, i64) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
// CHECK:   llvm.func @fetch_run_ctx(!llvm.ptr<i8>, i64) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
// CHECK:   llvm.func @_mlir__mlir_ciface_okl_compute(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
// CHECK:     %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
// CHECK:     %1 = llvm.mlir.constant(0 : index) : i64
// CHECK:     %2 = llvm.call @fetch_run_ctx(%arg0, %1) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     %3 = llvm.mlir.constant(1 : index) : i64
// CHECK:     %4 = llvm.call @fetch_run_ctx(%arg0, %3) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     %5 = llvm.mlir.constant(0 : index) : i64
// CHECK:     %6 = llvm.call @fetch_kernel(%arg0, %5) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     %7 = llvm.mlir.constant(1 : index) : i64
// CHECK:     %8 = llvm.call @fetch_kernel(%arg0, %7) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
// CHECK:     llvm.call @launch(%2, %6) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
// CHECK:     llvm.call @launch(%4, %8) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
// CHECK:     llvm.return
// CHECK:   }
// CHECK: }
module {
  llvm.func @_mlir__mlir_ciface_okl_compute(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
    %1 = "okl.fetch_run_ctx"(%0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %2 = "okl.fetch_run_ctx"(%0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %3 = "okl.fetch_kernel"(%0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    %4 = "okl.fetch_kernel"(%0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    "okl.launch"(%1, %3) : (!okl.run_ctx, !okl.kernel) -> ()
    "okl.launch"(%2, %4) : (!okl.run_ctx, !okl.kernel) -> ()
    llvm.return
  }
}

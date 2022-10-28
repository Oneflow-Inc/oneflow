// RUN: oneflow-opt %s \
// RUN: -reconcile-unrealized-casts \
// RUN: | FileCheck %s

// CHECK-NOT:  builtin.unrealized_conversion_cast

module {
  llvm.func @launch(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  llvm.func @fetch_kernel(!llvm.ptr<i8>, i64) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  llvm.func @fetch_run_ctx(!llvm.ptr<i8>, i64) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  llvm.func @_mlir__mlir_ciface_okl_compute(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.call @fetch_run_ctx(%arg0, %1) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.call @fetch_run_ctx(%arg0, %3) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.call @fetch_kernel(%arg0, %5) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.call @fetch_kernel(%arg0, %7) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    llvm.call @launch(%2, %6) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    llvm.call @launch(%4, %8) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    llvm.return
  }
}

// RUN: oneflow-opt %s \
// RUN: -reconcile-unrealized-casts \
// RUN: | FileCheck %s

// CHECK-NOT:  builtin.unrealized_conversion_cast

module {
  llvm.func @okl_llvm_func(!llvm.ptr<i8>, i64) attributes {llvm.emit_c_interface}
  func.func @okl_func(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
    %1 = llvm.mlir.constant(0 : index) : i64
    llvm.call @okl_llvm_func(%arg0, %1) : (!llvm.ptr<i8>, i64) -> ()
    %2 = llvm.mlir.constant(1 : index) : i64
    llvm.call @okl_llvm_func(%arg0, %2) : (!llvm.ptr<i8>, i64) -> ()
    return
  }
}

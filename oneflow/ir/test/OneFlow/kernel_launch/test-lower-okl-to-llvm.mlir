// RUN: oneflow-opt %s \
// RUN: -lower-okl-to-llvm  -canonicalize | FileCheck %s

module {
//   func.func @relu2D0(%arg0: tensor<1xf32>) -> tensor<1xf32> attributes {compiled = "true"} {
//     %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
//     return %0 : tensor<1xf32>
//   }
//   func.func @wrap0(%arg0: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) attributes {compiled = "true", llvm.emit_c_interface} {
//     %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
//     %1 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>
//     return %0, %1 : tensor<1xf32>, tensor<1xf32>
//   }
  llvm.func @okl_func(%arg0: !llvm.ptr<i8>) {
    %0 = "okl.reg_ctx"() {mlir_assembly = "\22func.func\22() ({\0A^bb0(%arg0: tensor<1xf32>):\0A  %0 = \22oneflow.relu\22(%arg0) {device_name = [\22@0:0\22], device_tag = \22cpu\22, hierarchy = [1], op_name = \22relu-0\22, scope_symbol_id = 12 : i64} : (tensor<1xf32>) -> tensor<1xf32>\0A  \22func.return\22(%0) : (tensor<1xf32>) -> ()\0A}) {compiled = \22true\22, function_type = (tensor<1xf32>) -> tensor<1xf32>, sym_name = \22relu2D0\22} : () -> ()"} : () -> !llvm.ptr<i8>
    %1 = "okl.run_ctx"(%0, %arg0) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    %2 = "okl.kernel"(%0) {op_type_name = "relu-0"} : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    "okl.launch"(%0, %1, %2) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    llvm.return
  }
}

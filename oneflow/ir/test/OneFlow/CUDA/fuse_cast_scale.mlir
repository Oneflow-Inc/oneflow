// RUN: oneflow-opt %s \
// RUN: -convert-linalg-to-loops \
// RUN: -convert-scf-to-std \
// RUN: -convert-linalg-to-llvm \
// RUN: -convert-memref-to-llvm \
// RUN: -convert-std-to-llvm \
// RUN: | oneflow-runner \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void

module  {
  func @Cast_289__FUSE__ScalarMulByTensor_290(%arg0: tensor<96x96xi64>, %arg1: tensor<1xf32>) -> tensor<96x96xf32> attributes {llvm.emit_c_interface} {
    %0 = "oneflow.cast"(%arg0) {device_name = ["@0:0"], device_tag = "gpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_289", output_lbns = ["Cast_289/out_0"], scope_symbol_id = 4611686018427478014 : i64} : (tensor<96x96xi64>) -> tensor<96x96xf32>
    %1 = "oneflow.scalar_mul_by_tensor"(%0, %arg1) {device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], op_name = "ScalarMulByTensor_290", output_lbns = ["ScalarMulByTensor_290/y_0"], scope_symbol_id = 4611686018427478014 : i64} : (tensor<96x96xf32>, tensor<1xf32>) -> tensor<96x96xf32>
    return %1 : tensor<96x96xf32>
  }
}

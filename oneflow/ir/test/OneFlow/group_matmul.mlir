// RUN: oneflow-opt %s \
// RUN: -group-matmul -canonicalize | FileCheck %s
module  {
  // CHECK-LABEL: func.func
  func.func @Cast_1__FUSE__ScalarMulByTensor_2(%x: tensor<2x320xf16>, %weight1: tensor<1280x320xf16>, %weight2: tensor<1280x320xf16>) -> (tensor<2x1280xf16>, tensor<2x1280xf16>) {
     %1 = "oneflow.matmul"(%x, %weight1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %2 = "oneflow.matmul"(%x, %weight2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
    return %1, %2 : tensor<2x1280xf16>, tensor<2x1280xf16>
  }
  //  CHECK: %0:3 = "oneflow.grouped_matmul_bias"(%arg0, %arg0, %arg0, %arg2, %arg2, %arg1)
}

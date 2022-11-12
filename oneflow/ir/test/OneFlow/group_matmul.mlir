// RUN: oneflow-opt %s \
// RUN: -group-matmul --mlir-print-ir-before-all --mlir-print-ir-after-all  | FileCheck %s
module  {
  // CHECK-LABEL: func.func
  func.func @no_bias(%x: tensor<2x320xf16>, %weight1: tensor<1280x320xf16>, %weight2: tensor<1280x320xf16>) -> (tensor<2x1280xf16>, tensor<2x1280xf16>) {
     %1 = "oneflow.matmul"(%x, %weight1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %2 = "oneflow.matmul"(%x, %weight2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
    return %1, %2 : tensor<2x1280xf16>, tensor<2x1280xf16>
    // CHECK: %0:2 = "oneflow.grouped_matmul_bias"(%arg0, %arg0, %arg2, %arg1)
    // CHECK: return %0#1, %0#0
  }

  // CHECK-LABEL: func.func
  func.func @with_bias(%x: tensor<2x320xf16>, %weight1: tensor<1280x320xf16>, %weight2: tensor<1280x320xf16>, %bias1: tensor<1280xf16>, %bias2: tensor<1280xf16>) -> (tensor<2x1280xf16>, tensor<2x1280xf16>) {
     %1 = "oneflow.matmul"(%x, %weight1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %r1 = "oneflow.bias_add"(%1, %bias1) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<2x1280xf16>, tensor<1280xf16>) -> tensor<2x1280xf16>
     %2 = "oneflow.matmul"(%x, %weight2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %r2 = "oneflow.bias_add"(%2, %bias2) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<2x1280xf16>, tensor<1280xf16>) -> tensor<2x1280xf16>
    return %r1, %r2 : tensor<2x1280xf16>, tensor<2x1280xf16>
    // CHECK: %0:2 = "oneflow.grouped_matmul_bias"(%arg0, %arg0, %arg2, %arg1, %arg4, %arg3)
    // CHECK: return %0#1, %0#0
  }


  // CHECK-LABEL: func.func
  func.func @mixed(%x: tensor<2x320xf16>, %weight1: tensor<1280x320xf16>, %weight2: tensor<1280x320xf16>, %bias1: tensor<1280xf16>, %bias2: tensor<1280xf16>) -> (tensor<2x1280xf16>, tensor<2x1280xf16>, tensor<2x1280xf16>, tensor<2x1280xf16>) {
     %1 = "oneflow.matmul"(%x, %weight1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %r1 = "oneflow.bias_add"(%1, %bias1) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<2x1280xf16>, tensor<1280xf16>) -> tensor<2x1280xf16>
     %2 = "oneflow.matmul"(%x, %weight2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %r2 = "oneflow.bias_add"(%2, %bias2) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<2x1280xf16>, tensor<1280xf16>) -> tensor<2x1280xf16>
     %m1 = "oneflow.matmul"(%x, %weight1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %m2 = "oneflow.matmul"(%x, %weight2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
    return %r1, %r2, %m1, %m2: tensor<2x1280xf16>, tensor<2x1280xf16>, tensor<2x1280xf16>, tensor<2x1280xf16>
    // CHECK: %0:2 = "oneflow.grouped_matmul_bias"(%arg0, %arg0, %arg2, %arg1, %arg4, %arg3)
    // CHECK: %1:2 = "oneflow.grouped_matmul_bias"(%arg0, %arg0, %arg2, %arg1)
    // CHECK: return %0#1, %0#0, %1#1, %1#0
  }
}

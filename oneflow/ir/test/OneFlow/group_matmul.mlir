// RUN: oneflow-opt %s \
// RUN: -group-matmul | FileCheck %s
module  {
  // CHECK-LABEL: func.func
  func.func @no_bias(%x: tensor<2x320xf16>, %weight1: tensor<1280x320xf16>, %weight2: tensor<1280x320xf16>) -> (tensor<2x1280xf16>, tensor<2x1280xf16>) {
     %1 = "oneflow.matmul"(%x, %weight1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %2 = "oneflow.matmul"(%x, %weight2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
    return %1, %2 : tensor<2x1280xf16>, tensor<2x1280xf16>
    // CHECK: @no_bias(%[[X:[a-zA-Z0-9_]+]]: tensor<2x320xf16>, %[[WEIGHT1:[a-zA-Z0-9_]+]]: tensor<1280x320xf16>, %[[WEIGHT2:[a-zA-Z0-9_]+]]: tensor<1280x320xf16>) -> (tensor<2x1280xf16>, tensor<2x1280xf16>)
    // CHECK: %[[OUT:[a-zA-Z0-9_]+]]:2 = "oneflow.grouped_matmul_bias"(%[[X]], %[[X]], %[[WEIGHT2]], %[[WEIGHT1]])
    // CHECK: return %[[OUT]]#1, %[[OUT]]#0
  }

  // CHECK-LABEL: func.func
  func.func @with_bias(%x: tensor<2x320xf16>, %weight1: tensor<1280x320xf16>, %weight2: tensor<1280x320xf16>, %bias1: tensor<1280xf16>, %bias2: tensor<1280xf16>) -> (tensor<2x1280xf16>, tensor<2x1280xf16>) {
     %1 = "oneflow.matmul"(%x, %weight1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %r1 = "oneflow.bias_add"(%1, %bias1) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<2x1280xf16>, tensor<1280xf16>) -> tensor<2x1280xf16>
     %2 = "oneflow.matmul"(%x, %weight2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %r2 = "oneflow.bias_add"(%2, %bias2) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<2x1280xf16>, tensor<1280xf16>) -> tensor<2x1280xf16>
    return %r1, %r2 : tensor<2x1280xf16>, tensor<2x1280xf16>
    // CHECK: @with_bias(%[[X:[a-zA-Z0-9_]+]]: tensor<2x320xf16>, %[[WEIGHT1:[a-zA-Z0-9_]+]]: tensor<1280x320xf16>, %[[WEIGHT2:[a-zA-Z0-9_]+]]: tensor<1280x320xf16>, %[[BIAS1:[a-zA-Z0-9_]+]]: tensor<1280xf16>, %[[BIAS2:[a-zA-Z0-9_]+]]: tensor<1280xf16>)
    // CHECK: %[[OUT:[a-zA-Z0-9_]+]]:2 = "oneflow.grouped_matmul_bias"(%[[X]], %[[X]], %[[WEIGHT2]], %[[WEIGHT1:[a-zA-Z0-9_]+]], %[[BIAS2:[a-zA-Z0-9_]+]], %[[BIAS1:[a-zA-Z0-9_]+]])
    // CHECK: return %[[OUT]]#1, %[[OUT]]#0
  }

  // CHECK-LABEL: func.func
  func.func @with_broadcast_add(%x: tensor<2x320xf16>, %weight1: tensor<1280x320xf16>, %weight2: tensor<1280x320xf16>, %bias1: tensor<1280xf16>, %bias2: tensor<1280xf16>) -> (tensor<2x1280xf16>, tensor<2x1280xf16>) {
     %1 = "oneflow.matmul"(%x, %weight1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %r1 = "oneflow.broadcast_add"(%1, %bias1) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<2x1280xf16>, tensor<1280xf16>) -> tensor<2x1280xf16>
     %2 = "oneflow.matmul"(%x, %weight2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %r2 = "oneflow.broadcast_add"(%2, %bias2) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<2x1280xf16>, tensor<1280xf16>) -> tensor<2x1280xf16>
    return %r1, %r2 : tensor<2x1280xf16>, tensor<2x1280xf16>
    // CHECK: @with_broadcast_add(%[[X:[a-zA-Z0-9_]+]]: tensor<2x320xf16>, %[[WEIGHT1:[a-zA-Z0-9_]+]]: tensor<1280x320xf16>, %[[WEIGHT2:[a-zA-Z0-9_]+]]: tensor<1280x320xf16>, %[[BIAS1:[a-zA-Z0-9_]+]]: tensor<1280xf16>, %[[BIAS2:[a-zA-Z0-9_]+]]: tensor<1280xf16>)
    // CHECK: %[[OUT:[a-zA-Z0-9_]+]]:2 = "oneflow.grouped_matmul_bias"(%[[X]], %[[X]], %[[WEIGHT2]], %[[WEIGHT1:[a-zA-Z0-9_]+]], %[[BIAS2:[a-zA-Z0-9_]+]], %[[BIAS1:[a-zA-Z0-9_]+]])
    // CHECK: return %[[OUT]]#1, %[[OUT]]#0
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
    // CHECK: @mixed(%[[X:[a-zA-Z0-9_]+]]: tensor<2x320xf16>, %[[WEIGHT1:[a-zA-Z0-9_]+]]: tensor<1280x320xf16>, %[[WEIGHT2:[a-zA-Z0-9_]+]]: tensor<1280x320xf16>, %[[BIAS1:[a-zA-Z0-9_]+]]: tensor<1280xf16>, %[[BIAS2:[a-zA-Z0-9_]+]]: tensor<1280xf16>)
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]]:2 = "oneflow.grouped_matmul_bias"(%[[X]], %[[X]], %[[WEIGHT2]], %[[WEIGHT1:[a-zA-Z0-9_]+]], %[[BIAS2:[a-zA-Z0-9_]+]], %[[BIAS1:[a-zA-Z0-9_]+]])
    // CHECK: %[[OUT1:[a-zA-Z0-9_]+]]:2 = "oneflow.grouped_matmul_bias"(%[[X]], %[[X]], %[[WEIGHT2]], %[[WEIGHT1]])
    // CHECK: return %[[OUT0]]#1, %[[OUT0]]#0, %[[OUT1]]#1, %[[OUT1]]#0
  }

  // CHECK-LABEL: func.func
  func.func @left_alone(%x: tensor<2x320xf16>, %weight1: tensor<1280x320xf16>, %weight2: tensor<1280x320xf16>, %bias1: tensor<1280xf16>, %bias2: tensor<1280xf16>) -> (tensor<2x1280xf16>, tensor<2x1280xf16>, tensor<2x1280xf16>) {
     %1 = "oneflow.matmul"(%x, %weight1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %r1 = "oneflow.bias_add"(%1, %bias1) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<2x1280xf16>, tensor<1280xf16>) -> tensor<2x1280xf16>
     %2 = "oneflow.matmul"(%x, %weight2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
     %r2 = "oneflow.bias_add"(%2, %bias2) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<2x1280xf16>, tensor<1280xf16>) -> tensor<2x1280xf16>
     %m1 = "oneflow.matmul"(%x, %weight1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<2x320xf16>, tensor<1280x320xf16>) -> tensor<2x1280xf16>
    return %r1, %r2, %m1: tensor<2x1280xf16>, tensor<2x1280xf16>, tensor<2x1280xf16>
    // CHECK: @left_alone(%[[X:[a-zA-Z0-9_]+]]: tensor<2x320xf16>, %[[WEIGHT1:[a-zA-Z0-9_]+]]: tensor<1280x320xf16>, %[[WEIGHT2:[a-zA-Z0-9_]+]]: tensor<1280x320xf16>, %[[BIAS1:[a-zA-Z0-9_]+]]: tensor<1280xf16>, %[[BIAS2:[a-zA-Z0-9_]+]]: tensor<1280xf16>)
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]]:2 = "oneflow.grouped_matmul_bias"(%[[X]], %[[X]], %[[WEIGHT2]], %[[WEIGHT1:[a-zA-Z0-9_]+]], %[[BIAS2:[a-zA-Z0-9_]+]], %[[BIAS1:[a-zA-Z0-9_]+]])
    // CHECK: %[[OUT1:[a-zA-Z0-9_]+]] = "oneflow.matmul"(%arg0, %arg1)
    // CHECK: return %[[OUT0]]#1, %[[OUT0]]#0, %[[OUT1]]
  }
  func.func @f_broadcast_matmul(%x: tensor<2x4096x320xf16>, %w1: tensor<320x320xf16>, %w2: tensor<320x320xf16>, %w3: tensor<320x320xf16>) -> (tensor<2x4096x320xf16>, tensor<2x4096x320xf16>, tensor<2x4096x320xf16>) {
    %matmul1 = "oneflow.broadcast_matmul"(%x, %w1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q-broadcast_matmul-16315", scope_symbol_id = 5497 : i64, transpose_a = false, transpose_b = true} : (tensor<2x4096x320xf16>, tensor<320x320xf16>) -> tensor<2x4096x320xf16>
    %matmul2 = "oneflow.broadcast_matmul"(%x, %w2) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k-broadcast_matmul-16316", scope_symbol_id = 5505 : i64, transpose_a = false, transpose_b = true} : (tensor<2x4096x320xf16>, tensor<320x320xf16>) -> tensor<2x4096x320xf16>
    %matmul3 = "oneflow.broadcast_matmul"(%x, %w3) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v-broadcast_matmul-16317", scope_symbol_id = 5513 : i64, transpose_a = false, transpose_b = true} : (tensor<2x4096x320xf16>, tensor<320x320xf16>) -> tensor<2x4096x320xf16>
    return %matmul1, %matmul2, %matmul3 : tensor<2x4096x320xf16>, tensor<2x4096x320xf16>, tensor<2x4096x320xf16>
    // CHECK: @f_broadcast_matmul(%[[X:[a-zA-Z0-9_]+]]: tensor<2x4096x320xf16>, %[[WEIGHT1:[a-zA-Z0-9_]+]]: tensor<320x320xf16>, %[[WEIGHT2:[a-zA-Z0-9_]+]]: tensor<320x320xf16>, %[[WEIGHT3:[a-zA-Z0-9_]+]]: tensor<320x320xf16>)
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]]:3 = "oneflow.grouped_matmul_bias"(%[[X]], %[[X]], %[[X]], %[[WEIGHT3]], %[[WEIGHT2]], %[[WEIGHT1]])
    // CHECK: return %[[OUT0]]#2, %[[OUT0]]#1, %[[OUT0]]#0
  }

  func.func @test_fused_matmul_bias_graph(%x: tensor<8x9xf64>, %w: tensor<10x9xf64>, %bias: tensor<10xf64>) -> (tensor<8x10xf64>, tensor<8x10xf64>) {
    %y0 = "oneflow.fused_matmul_bias"(%x, %w, %bias) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "fused_matmul_bias-0", scope_symbol_id = 12 : i64} : (tensor<8x9xf64>, tensor<10x9xf64>, tensor<10xf64>) -> tensor<8x10xf64>
    %y1 = "oneflow.fused_matmul_bias"(%x, %w, %bias) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "fused_matmul_bias-0", scope_symbol_id = 12 : i64} : (tensor<8x9xf64>, tensor<10x9xf64>, tensor<10xf64>) -> tensor<8x10xf64>
    return %y0, %y1 : tensor<8x10xf64>, tensor<8x10xf64>
    // CHECK: @test_fused_matmul_bias_graph(%[[X:[a-zA-Z0-9_]+]]: tensor<8x9xf64>, %[[W:[a-zA-Z0-9_]+]]: tensor<10x9xf64>, %[[BIAS:[a-zA-Z0-9_]+]]: tensor<10xf64>)
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]]:2 = "oneflow.grouped_matmul_bias"(%[[X]], %[[X]], %[[W]], %[[W]], %[[BIAS]], %[[BIAS]])
    // CHECK: return %[[OUT0]]#1, %[[OUT0]]#0
  }

  func.func @test_fused_matmul_bias_graph_mixed(%x: tensor<8x9xf64>, %w: tensor<10x9xf64>, %bias: tensor<10xf64>, %w1: tensor<10x9xf64>, %bias1: tensor<10xf64>) -> (tensor<8x10xf64>, tensor<8x10xf64>, tensor<8x10xf64>) {
    %y0 = "oneflow.fused_matmul_bias"(%x, %w, %bias) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "fused_matmul_bias-0", scope_symbol_id = 12 : i64} : (tensor<8x9xf64>, tensor<10x9xf64>, tensor<10xf64>) -> tensor<8x10xf64>
    %y1 = "oneflow.fused_matmul_bias"(%x, %w, %bias) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "fused_matmul_bias-0", scope_symbol_id = 12 : i64} : (tensor<8x9xf64>, tensor<10x9xf64>, tensor<10xf64>) -> tensor<8x10xf64>
    %matmul = "oneflow.matmul"(%x, %w1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-matmul-20", scope_symbol_id = 90 : i64, transpose_a = false, transpose_b = true} : (tensor<8x9xf64>, tensor<10x9xf64>) ->  tensor<8x10xf64>
    %bias_add = "oneflow.bias_add"(%matmul, %bias1) {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet.time_embedding.linear_1-bias_add-21", scope_symbol_id = 90 : i64} : (tensor<8x10xf64>, tensor<10xf64>) -> tensor<8x10xf64>
    return %y0, %y1, %bias_add : tensor<8x10xf64>, tensor<8x10xf64>, tensor<8x10xf64>
    // CHECK: @test_fused_matmul_bias_graph_mixed(%[[X:[a-zA-Z0-9_]+]]: tensor<8x9xf64>, %[[W:[a-zA-Z0-9_]+]]: tensor<10x9xf64>, %[[BIAS:[a-zA-Z0-9_]+]]: tensor<10xf64>, %[[W1:[a-zA-Z0-9_]+]]: tensor<10x9xf64>, %[[BIAS1:[a-zA-Z0-9_]+]]: tensor<10xf64>)
    // CHECK: %[[OUT0:[a-zA-Z0-9_]+]]:3 = "oneflow.grouped_matmul_bias"(%[[X]], %[[X]], %[[X]], %[[W1]], %[[W]], %[[W]], %[[BIAS1]], %[[BIAS]], %[[BIAS]])
    // CHECK: return %[[OUT0]]#2, %[[OUT0]]#1, %[[OUT0]]#0
  }
}

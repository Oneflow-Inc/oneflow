module  {
  func @Matmul_1(%arg0: tensor<1x20x30xf32>, %arg1: tensor<1x30x20xf32>) -> tensor<1x20x20xf32> {
    %0 = "oneflow.batch_matmul"(%arg0, %arg1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], op_name = "Matmul_1", output_lbns = ["Matmul_1/out_0"], scope_symbol_id = 4611686018427453438 : i64, transpose_a = false, transpose_b = false} : (tensor<1x20x30xf32>, tensor<1x30x20xf32>) -> tensor<1x20x20xf32>
    return %0 : tensor<1x20x20xf32>
  }
}

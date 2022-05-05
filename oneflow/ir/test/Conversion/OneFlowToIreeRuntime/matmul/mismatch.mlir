//RUN: oneflow-opt -lower-oneflow-to-tosa %s 2>&1 || true


 module  {
  func @main(%arg0: tensor<1x2xf32>, %arg1: tensor<2x1xi32>) -> tensor<1x1xf32> {
    %res = "oneflow.matmul"(%arg0, %arg1) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "matmul-439", output_lbns = ["matmul-439/out_0"], scope_symbol_id = 4611686018427416575 : i64, transpose_a = false, transpose_b = false} : (tensor<1x2xf32>, tensor<2x1xi32>) -> tensor<1x1xf32>
    return %res : tensor<1x1xf32>

  }
}


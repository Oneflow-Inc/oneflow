
// RUN: oneflow-opt -lower-oneflow-to-tosa -pass-pipeline="func.func(tosa-to-linalg)" -cse --linalg-fuse-elementwise-ops -linalg-bufferize -tensor-bufferize -func-bufferize -buffer-results-to-out-params -convert-linalg-to-loops -convert-scf-to-cf -convert-linalg-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -reconcile-unrealized-casts --print-after-all %s
// RUN: oneflow-opt -lower-oneflow-to-tosa -pass-pipeline="func.func(tosa-to-linalg)" -cse --linalg-fuse-elementwise-ops  -linalg-bufferize -tensor-bufferize -func-bufferize -buffer-results-to-out-params  -finalizing-bufferize -canonicalize %s

module  {
  func @ResnetGraph_0(%arg0: tensor<16x64x112x112xf32>) -> tensor<16x64x112x112xf32> {
    %1 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-7", output_lbns = ["relu-7/y_0"], scope_symbol_id = 4611686018427416575 : i64} : (tensor<16x64x112x112xf32>) -> tensor<16x64x112x112xf32>
    return %1 : tensor<16x64x112x112xf32>
  }
}

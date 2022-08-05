// RUN: oneflow-opt %s \
// RUN: -pass-pipeline="func.func(tosa-to-linalg)" -cse \
// RUN: --linalg-fuse-elementwise-ops -linalg-bufferize \
// RUN: -tensor-bufferize -func-bufferize -buffer-results-to-out-params \
// RUN: -convert-linalg-to-loops -convert-scf-to-cf -convert-linalg-to-llvm \
// RUN: -convert-func-to-llvm -convert-memref-to-llvm -reconcile-unrealized-casts --print-after-all \
// RUN: | oneflow-translate -mlir-to-llvmir

builtin.module {
  func.func @Graph_0(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "tosa.cast"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "tosa.reluN"(%0) {max_fp = 3.40282347E+38 : f32, max_int = 9223372036854775807 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %2 = "tosa.cast"(%1) : (tensor<2xf32>) -> tensor<2xf32>
    func.return %2 : tensor<2xf32>
  }
}

// RUN: oneflow-opt %s \
// RUN: -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" \
// RUN: | oneflow-opt -cse \
// RUN: --linalg-fuse-elementwise-ops -empty-tensor-to-alloc-tensor -linalg-bufferize \
// RUN: -tensor-bufferize -func-bufferize -buffer-results-to-out-params \
// RUN: -convert-linalg-to-loops -convert-math-to-libm -convert-math-to-llvm -convert-scf-to-cf -convert-linalg-to-llvm \
// RUN: -convert-func-to-llvm -finalize-memref-to-llvm -reconcile-unrealized-casts --print-after-all \
// RUN: | oneflow-translate -mlir-to-llvmir

builtin.module {
  func.func @Graph_0(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "tosa.cast"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "tosa.tanh"(%0) : (tensor<2xf32>) -> tensor<2xf32>
    %2 = "tosa.cast"(%1) : (tensor<2xf32>) -> tensor<2xf32>
    func.return %2 : tensor<2xf32>
  }
}

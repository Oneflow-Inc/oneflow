// RUN: oneflow-opt -lower-oneflow-to-func %s | FileCheck %s
// CHECK: return %arg0 : tensor<16xf32>

module {
  oneflow.job nested @ResnetGraph_0(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    oneflow.return %arg0 : tensor<16xf32>
  }
}

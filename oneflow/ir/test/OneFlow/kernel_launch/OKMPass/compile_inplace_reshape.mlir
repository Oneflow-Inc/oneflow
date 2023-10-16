// RUN: oneflow-opt %s --mlir-print-ir-after-all \
// RUN: -extract-okm-tensor \
// RUN: -wrap-okm-kernel \
// RUN: -opt-okm-memref \
// RUN: -convert-okm-to-okl \
// RUN: | FileCheck %s


module {
  func.func @_mlir_oneflow_subgraph0(%arg0: tensor<12xf16>) -> tensor<3x4xf16> {
    %1 = "oneflow.reshape"(%arg0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet-reshape-21", scope_symbol_id = 28 : i64, shape = [3 : si64, 4 : si64]} : (tensor<12xf16>) -> tensor<3x4xf16>
    func.return %1 : tensor<3x4xf16>
  }
}

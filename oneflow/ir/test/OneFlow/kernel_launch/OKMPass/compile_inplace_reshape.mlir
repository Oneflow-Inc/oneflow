// RUN: oneflow-opt %s --mlir-print-ir-after-all \
// RUN: -extract-okm-tensor \
// RUN: -wrap-okm-kernel \
// RUN: -opt-okm-memref \
// RUN: -convert-okm-to-okl \
// RUN: | FileCheck %s


module {
  func.func @okm_subgraph0(%arg0: tensor<24xi8>) -> tensor<3x4xf16> {
    %0 = "okm.arg_to_tensor"() {index = 0 : i32} : () -> tensor<24xi8>
    %1 = "oneflow.reshape"(%0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "unet-reshape-21", scope_symbol_id = 28 : i64, shape = [3 : si64, 4 : si64]} : (tensor<24xi8>) -> tensor<3x4xf16>
    %2 = "okm.tensor_to_ret"(%1) {index = 0 : i32} : (tensor<3x4xf16>) -> tensor<3x4xf16>
    func.return %2 : tensor<3x4xf16>
  }
}

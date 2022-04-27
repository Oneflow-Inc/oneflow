// oneflow-opt -lower-oneflow-to-func %s | FileCheck %s

module {
  oneflow.job nested @ResnetGraph_0(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %output = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], nd_sbp = ["B"], op_name = "FreeEagerTensor-1", output_lbns = ["FreeEagerTensor-1/out"], scope_symbol_id = 4611686018427416575 : i64, shape = [64 : si64, 3 : si64, 7 : si64, 7 : si64], trainable = false} : () -> tensor<64x3x7x7xf32>
    oneflow.return %arg0 : tensor<16xf32>
  }
}

// RUN: oneflow-opt -lower-oneflow-to-tosa %s
module {
  oneflow.job @GraphModule_0() -> tensor<64x3x7x7xf32> {
   %0 = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu",
    hierarchy = [1], nd_sbp = ["B"], op_name = "fw.model.conv1.weight", output_lbns = ["fw.model.conv1.weight/out"],
     scope_symbol_id = 4611686018427432959 : i64, shape = [64 : si64, 3 : si64, 7 : si64, 7 : si64]} :
     () -> tensor<64x3x7x7xf32>
    oneflow.return %0 : tensor<64x3x7x7xf32>
  }
}

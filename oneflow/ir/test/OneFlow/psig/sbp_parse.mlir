// RUN: oneflow-opt %s \
// RUN: -split-input-file \
// RUN: -verify-diagnostics -o -


// CHECK-LABEL: test_func
module {
  oneflow.job @test_func(){
    // CHECK: nd_sbp = #sbp.parallel_signature<[] -> [[#sbp.b, #sbp.s<0>]]>
    %output = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0", "@1:1"], device_tag = "cuda", hierarchy = [2, 1], nd_sbp = #sbp.parallel_signature<[] -> [[#sbp.b, #sbp.s<0>]]>, op_name = "net-FreeEagerTensor-1", output_lbns = ["net-FreeEagerTensor-1/out"], scope_symbol_id = 14 : i64, shape = [4 : si64, 5 : si64], trainable = false} : () -> tensor<4x5xf32>
    // CHECK: nd_sbp = #sbp.parallel_signature<[] -> [[#sbp.b, #sbp.p]]>
    %output_0 = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0", "@1:1"], device_tag = "cuda", hierarchy = [2, 1], nd_sbp = #sbp.parallel_signature<[] -> [[#sbp.b, #sbp.p]]>, op_name = "net-FreeEagerTensor-2", output_lbns = ["net-FreeEagerTensor-2/out"], scope_symbol_id = 14 : i64, shape = [5 : si64, 8 : si64], trainable = false} : () -> tensor<5x8xf32>
    oneflow.return
  }
}

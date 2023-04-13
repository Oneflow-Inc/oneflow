// RUN: oneflow-opt %s \
// RUN: -split-input-file \
// RUN: -verify-diagnostics -o - | FileCheck %s

// CHECK-LABEL: test_single
module {
  oneflow.job @test_single(){
// CHECK: parallel = #sbp.parallel<[] -> [#sbp.B, #sbp.S<0>]>
    %output = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0", "@1:1"], device_tag = "cuda", hierarchy = [2, 1], parallel = #sbp.parallel<[] -> [#sbp.B, #sbp.S<0>]>, op_name = "net-FreeEagerTensor-1", output_lbns = ["net-FreeEagerTensor-1/out"], scope_symbol_id = 14 : i64, shape = [4 : si64, 5 : si64], trainable = false} : () -> tensor<4x5xf32>
// CHECK: parallel = #sbp.parallel<[] -> [#sbp.B, #sbp.P]>
    %output_0 = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0", "@1:1"], device_tag = "cuda", hierarchy = [2, 1], parallel = #sbp.parallel<[] -> [#sbp.B, #sbp.P]>, op_name = "net-FreeEagerTensor-2", output_lbns = ["net-FreeEagerTensor-2/out"], scope_symbol_id = 14 : i64, shape = [5 : si64, 8 : si64], trainable = false} : () -> tensor<5x8xf32>
    oneflow.return
  }
}

// CHECK-LABEL: test_nd
module {
  oneflow.job @test_nd(){
    // CHECK: #sbp.B, #sbp.S<0>
    %output = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0", "@1:1"], device_tag = "cuda", hierarchy = [2, 1], parallel = #sbp.parallel<[] -> [[#sbp.B, #sbp.S<0>]]>, op_name = "net-FreeEagerTensor-1", output_lbns = ["net-FreeEagerTensor-1/out"], scope_symbol_id = 14 : i64, shape = [4 : si64, 5 : si64], trainable = false} : () -> tensor<4x5xf32>
    // CHECK: [#sbp.B, #sbp.P]
    %output_0 = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0", "@1:1"], device_tag = "cuda", hierarchy = [2, 1], parallel = #sbp.parallel<[] -> [[#sbp.B, #sbp.P]]>, op_name = "net-FreeEagerTensor-2", output_lbns = ["net-FreeEagerTensor-2/out"], scope_symbol_id = 14 : i64, shape = [5 : si64, 8 : si64], trainable = false} : () -> tensor<5x8xf32>
    oneflow.return
  }
}

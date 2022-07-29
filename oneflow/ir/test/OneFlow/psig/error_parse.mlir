// RUN: not oneflow-opt %s \
// RUN: -split-input-file \
// RUN: -verify-diagnostics -o -  2>&1 | FileCheck  --check-prefix=CHECK_ERROR_1  %s

// CHECK_ERROR_1: unexpected error: failed to parse a sbp attribute here
module {
  oneflow.job @test_err(){
    %output_0 = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0", "@1:1"], device_tag = "cuda", hierarchy = [2, 1], parallel = #sbp.parallel<[] -> [[[]], "S(0)", #sbp.P]>, op_name = "net-FreeEagerTensor-2", output_lbns = ["net-FreeEagerTensor-2/out"], scope_symbol_id = 14 : i64, shape = [5 : si64, 8 : si64], trainable = false} : () -> tensor<5x8xf32>
    oneflow.return
  }
}

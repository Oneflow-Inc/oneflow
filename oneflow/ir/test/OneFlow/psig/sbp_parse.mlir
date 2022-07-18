// RUN: oneflow-opt %s \
// RUN: -split-input-file \
// RUN: -verify-diagnostics -o -


// CHECK-LABEL: test_func
func.func @test_func(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
    "builtin.unrealized_conversion_cast"() {parallel_signature = #sbp.sbp<"s(0)"> } : () -> f32
    func.return %arg0 : tensor<1xf32>
}

// RUN: oneflow-opt %s \
// RUN: -split-input-file \
// RUN: -verify-diagnostics -o - \
// RUN: | FileCheck %s


// CHECK-LABEL: test_func
func.func @test_func(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
    "builtin.unrealized_conversion_cast"() {parallel_signature = #sbp.parallel_signature<[#sbp.s<0>] -> [#sbp.s<0>]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signature = #sbp.parallel_signature<[#sbp.s<0>, #sbp.s<0>] -> [#sbp.s<0>]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signature = #sbp.parallel_signature<[[#sbp.s<0>, #sbp.p], #sbp.s<0>, #sbp.p] -> [[#sbp.s<0>, #sbp.p]]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signature = #sbp.parallel_signature<[[#sbp.s<0>, #sbp.p], #sbp.s<0>] -> [#sbp.s<0>]>} : () -> f32
    // CHECK: [#sbp.s<0>, #sbp.p], #sbp.s<0>] -> [#sbp.s<0>]
    "builtin.unrealized_conversion_cast"() {parallel_signature = #sbp.s<1>} : () -> f32
    // CHECK: builtin.unrealized_conversion_cast to f32 {parallel_signature = #sbp.s<1>}
    "builtin.unrealized_conversion_cast"() {parallel_signature = #sbp.b} : () -> f32
    // CHECK: builtin.unrealized_conversion_cast to f32 {parallel_signature = #sbp.b}
    "builtin.unrealized_conversion_cast"() {parallel_signature = #sbp.p} : () -> f32
    // CHECK: builtin.unrealized_conversion_cast to f32 {parallel_signature = #sbp.p}
    func.return %arg0 : tensor<1xf32>
    // CHECK: return [[V0:%.+]] : tensor<1xf32>
}

// RUN: oneflow-opt %s \
// RUN: -split-input-file \
// RUN: -verify-diagnostics -o - \
// RUN: | FileCheck %s


// CHECK-LABEL: test_func
func.func @test_func(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
    "builtin.unrealized_conversion_cast"() {parallel_signature = #oneflow.parallel_signature<[#oneflow.s<0>] -> [#oneflow.s<0>]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signature = #oneflow.parallel_signature<[#oneflow.s<0>, #oneflow.s<0>] -> [#oneflow.s<0>]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signature = #oneflow.parallel_signature<[[#oneflow.s<0>, #oneflow.p], #oneflow.s<0>, #oneflow.p] -> [[#oneflow.s<0>, #oneflow.p]]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signature = #oneflow.parallel_signature<[[#oneflow.s<0>, #oneflow.p], #oneflow.s<0>] -> [#oneflow.s<0>]>} : () -> f32
    // CHECK: [#oneflow.s<0>, #oneflow.p], #oneflow.s<0>] -> [#oneflow.s<0>]
    "builtin.unrealized_conversion_cast"() {parallel_signature = #oneflow.s<1>} : () -> f32
    // CHECK: builtin.unrealized_conversion_cast to f32 {parallel_signature = #oneflow.s<1>}
    "builtin.unrealized_conversion_cast"() {parallel_signature = #oneflow.b} : () -> f32
    // CHECK: builtin.unrealized_conversion_cast to f32 {parallel_signature = #oneflow.b}
    "builtin.unrealized_conversion_cast"() {parallel_signature = #oneflow.p} : () -> f32
    // CHECK: builtin.unrealized_conversion_cast to f32 {parallel_signature = #oneflow.p}
    func.return %arg0 : tensor<1xf32>
    // CHECK: return [[V0:%.+]] : tensor<1xf32>
}

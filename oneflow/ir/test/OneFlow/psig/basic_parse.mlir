// RUN: oneflow-opt %s \
// RUN: -split-input-file \
// RUN: -verify-diagnostics -o - \
// RUN: | FileCheck %s


// CHECK-LABEL: test_func
func.func @test_func(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
    // "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.s<0>} : () -> f32
    // "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.psig<[#oneflow.s<0>], [#oneflow.s<0>]>} : () -> f32
    // "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.psig<[#oneflow.s<0>, #oneflow.s<0>], [#oneflow.s<0>]>} : () -> f32
    // "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.psig<[[#oneflow.s<0>, #oneflow.p], #oneflow.s<0>, #oneflow.p], [[#oneflow.s<0>, #oneflow.p]]>} : () -> f32
    // "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.psig<[[#oneflow.s<0>, #oneflow.p], #oneflow.s<0>], [#oneflow.s<0>]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.s<1>} : () -> f32
    // CHECK: builtin.unrealized_conversion_cast to f32 {parallel_signaure = #oneflow.s<1>}
    "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.b} : () -> f32
    // CHECK: builtin.unrealized_conversion_cast to f32 {parallel_signaure = #oneflow.b}
    "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.p} : () -> f32
    // CHECK: builtin.unrealized_conversion_cast to f32 {parallel_signaure = #oneflow.p}
    func.return %arg0 : tensor<1xf32>
    // CHECK: return [[V0:%.+]] : tensor<1xf32>
}

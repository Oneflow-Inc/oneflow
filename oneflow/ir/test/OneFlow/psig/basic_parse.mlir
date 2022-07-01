// RUN: oneflow-opt %s \
// RUN: -split-input-file \
// RUN: -verify-diagnostics -o - \
// RUN: | FileCheck %s


// CHECK-LABEL: test_func
// CHECK: return [[V0:%.+]] : tensor<1xf32>
oneflow.job @test_func(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
    "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.s<0>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.psig<[#oneflow.s<0>], [#oneflow.s<0>]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.psig<[#oneflow.s<0>, #oneflow.s<0>], [#oneflow.s<0>]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.psig<[[#oneflow.s<0>, #oneflow.p], #oneflow.s<0>, #oneflow.p], [[#oneflow.s<0>, #oneflow.p]]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {parallel_signaure = #oneflow.psig<[[#oneflow.s<0>, #oneflow.p], #oneflow.s<0>], [#oneflow.s<0>]>} : () -> f32
    "builtin.unrealized_conversion_cast"() {a = #oneflow.s<1>} : () -> f32
    "builtin.unrealized_conversion_cast"() {a = #oneflow.b} : () -> f32
    "builtin.unrealized_conversion_cast"() {b = #oneflow.p} : () -> f32
    oneflow.return %arg0 : tensor<1xf32>
}

// RUN: python3 %s.py | FileCheck %s
// CHECK: module
// CHECK: %0 = "oneflow.cast"
// CHECK: %1 = "oneflow.scalar_mul_by_tensor"

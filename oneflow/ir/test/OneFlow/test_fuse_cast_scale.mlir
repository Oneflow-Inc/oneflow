// RUN: python3 %s.py | FileCheck %s
// TODO: print IR before fuser pass and after (CHECK-LABEL)
// CHECK: module
// CHECK: %0 = "oneflow.mlir_jit"
// CHECK: func @

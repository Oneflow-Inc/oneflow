// RUN: oneflow-opt --oneflow-transform-dialect-interpreter %s -split-input-file -verify-diagnostics | FileCheck %s

// Test One-Shot Bufferize.

transform.sequence failures(propagate) {
^bb0(%arg1: !pdl.operation):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %1 = transform.bufferization.one_shot_bufferize %0 : (!pdl.operation) -> !pdl.operation
}

// CHECK-LABEL: func @test_function(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>
func.func @test_function(%A : tensor<?xf32>, %v : vector<4xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index

  // CHECK: %[[A_memref:.*]] = bufferization.to_memref %[[A]]
  // CHECK: %[[dim:.*]] = memref.dim %[[A_memref]]
  // CHECK: %[[alloc:.*]] = memref.alloc(%[[dim]])
  // CHECK: memref.copy %[[A_memref]], %[[alloc]]
  // CHECK: vector.transfer_write %{{.*}}, %[[alloc]]
  // CHECK: %[[res_tensor:.*]] = bufferization.to_tensor %[[alloc]]
  %0 = vector.transfer_write %v, %A[%c0] : vector<4xf32>, tensor<?xf32>

  // CHECK: memref.dealloc %[[alloc]]
  // CHECK: return %[[res_tensor]]
  return %0 : tensor<?xf32>
}
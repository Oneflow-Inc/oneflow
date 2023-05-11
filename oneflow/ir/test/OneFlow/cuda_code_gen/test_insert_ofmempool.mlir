// RUN: oneflow-opt %s -insert-ofmempool | FileCheck %s

#map = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module attributes {gpu.container_module} {
  // CHECK: func.func @JITOpGenerated0(%[[ARG0:[a-zA-Z0-9_]+]]: memref<512xi8>
  func.func @JITOpGenerated0(%arg0: memref<1xf32>, %arg1: memref<5xi64>, %arg2: memref<5xf32>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    // CHECK-NOT: memref.alloc() : memref<512xi8>
    %alloc = memref.alloc() : memref<512xi8>
    // CHECK: memref.view %[[ARG0]]
    %view = memref.view %alloc[%c0][] : memref<512xi8> to memref<5xf32>
    %collapse_shape = memref.collapse_shape %arg0 [] : memref<1xf32> into memref<f32>
    gpu.launch_func  @JITOpGenerated0_kernel::@JITOpGenerated0_kernel blocks in (%c5, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg1 : memref<5xi64>, %view : memref<5xf32>)
    gpu.launch_func  @JITOpGenerated0_kernel_0::@JITOpGenerated0_kernel blocks in (%c5, %c1, %c1) threads in (%c1, %c1, %c1) args(%view : memref<5xf32>, %collapse_shape : memref<f32>, %arg2 : memref<5xf32>)
    return
  }
  gpu.module @JITOpGenerated0_kernel {
    gpu.func @JITOpGenerated0_kernel(%arg0: memref<5xi64>, %arg1: memref<5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %0 = gpu.block_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %1 = affine.apply #map(%0)[%c1, %c0]
      %2 = memref.load %arg0[%1] : memref<5xi64>
      %3 = arith.sitofp %2 : i64 to f32
      memref.store %3, %arg1[%1] : memref<5xf32>
      gpu.return
    }
  }
  gpu.module @JITOpGenerated0_kernel_0 {
    gpu.func @JITOpGenerated0_kernel(%arg0: memref<5xf32>, %arg1: memref<f32>, %arg2: memref<5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %0 = gpu.block_id  x
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %1 = affine.apply #map(%0)[%c1, %c0]
      %2 = memref.load %arg0[%1] : memref<5xf32>
      %3 = memref.load %arg1[] : memref<f32>
      %4 = arith.mulf %2, %3 : f32
      memref.store %4, %arg2[%1] : memref<5xf32>
      gpu.return
    }
  }
}
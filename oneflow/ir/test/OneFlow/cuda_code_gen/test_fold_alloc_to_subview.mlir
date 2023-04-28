// RUN: oneflow-opt %s -fold-alloc-to-subview
#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module attributes {gpu.container_module} {
  func.func @JITOpGenerated0(%arg0: memref<1xf32>, %arg1: memref<5xi64>, %arg2: memref<5xf32>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %collapse_shape = memref.collapse_shape %arg0 [] : memref<1xf32> into memref<f32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<5xf32>
    // CHECK-NOT: %alloc = memref.alloc() {alignment = 64 : i64} : memref<5xf32>
    // CHECK: memref.alloc() : memref<512xi8>
    // CHECK: memref.view
    %c1_0 = arith.constant 1 : index
    %0 = affine.apply #map(%c5)[%c0, %c1]
    gpu.launch_func  @JITOpGenerated0_kernel::@JITOpGenerated0_kernel blocks in (%0, %c1_0, %c1_0) threads in (%c1_0, %c1_0, %c1_0) args(%arg1 : memref<5xi64>, %alloc : memref<5xf32>)
    %c1_2 = arith.constant 1 : index
    %1 = affine.apply #map(%c5)[%c0, %c1]
    gpu.launch_func  @JITOpGenerated0_kernel_0::@JITOpGenerated0_kernel blocks in (%1, %c1_2, %c1_2) threads in (%c1_2, %c1_2, %c1_2) args(%alloc : memref<5xf32>, %collapse_shape : memref<f32>, %arg2 : memref<5xf32>)
    return
  }
  gpu.module @JITOpGenerated0_kernel {
    gpu.func @JITOpGenerated0_kernel(%arg0: memref<5xi64>, %arg1: memref<5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %12 = affine.apply #map1(%0)[%c1, %c0]
      %13 = memref.load %arg0[%12] : memref<5xi64>
      %14 = arith.sitofp %13 : i64 to f32
      memref.store %14, %arg1[%12] : memref<5xf32>
      gpu.return
    }
  }
  gpu.module @JITOpGenerated0_kernel_0 {
    gpu.func @JITOpGenerated0_kernel(%arg0: memref<5xf32>, %arg1: memref<f32>, %arg2: memref<5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %12 = affine.apply #map1(%0)[%c1, %c0]
      %13 = memref.load %arg0[%12] : memref<5xf32>
      %14 = memref.load %arg1[] : memref<f32>
      %15 = arith.mulf %13, %14 : f32
      memref.store %15, %arg2[%12] : memref<5xf32>
      gpu.return
    }
  }
}

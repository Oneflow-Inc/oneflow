// RUN: oneflow-opt %s -split-input-file -oneflow-post-transform-checks -verify-diagnostics

// -----

module  {
// expected-error @+1 {{find illegal ops in current func.func body}}
  func.func @Cast_1__FUSE__ScalarMulByTensor_2(%arg0: tensor<96x96xi64>, %arg1: tensor<1xf32>) -> tensor<96x96xf32> {
// expected-error @+1 {{failed to legalize operation 'oneflow.cast' that was explicitly marked illegal}}
    %0 = "oneflow.cast"(%arg0) {device_name = ["0:0"], device_tag = "cpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_1", op_type_name = "cast", scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xi64>) -> tensor<96x96xf32>
    %1 = "oneflow.scalar_mul_by_tensor"(%0, %arg1) {device_name = ["0:0"], device_tag = "cpu", hierarchy = [1], op_name = "ScalarMulByTensor_2", op_type_name = "scalar_mul_by_tensor", scope_symbol_id = 4611686018427416574 : i64} : (tensor<96x96xf32>, tensor<1xf32>) -> tensor<96x96xf32>
    return %1 : tensor<96x96xf32>
  }
}

// -----

module attributes {gpu.container_module} {
// expected-error @+1 {{find illegal ops in current func.func body}}
  func.func @JITOpGenerated0(%arg0: memref<5xf32>, %arg1: memref<5xf32>) attributes {llvm.emit_c_interface, oneflow.gpu} {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -1.000000e+30 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc[] : memref<f32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %alloc, %alloc_1 : memref<f32> to memref<f32>
// expected-error @+1 {{failed to legalize operation 'scf.for' that was explicitly marked illegal}}
    scf.for %arg2 = %c0 to %c5 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<5xf32>
      %1 = memref.load %alloc_1[] : memref<f32>
      %2 = arith.maxf %0, %1 : f32
      memref.store %2, %alloc_1[] : memref<f32>
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<5xf32>
    gpu.launch_func  @JITOpGenerated0_kernel::@JITOpGenerated0_kernel blocks in (%c5, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<5xf32>, %alloc_1 : memref<f32>, %alloc_2 : memref<5xf32>)
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_3[] : memref<f32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %alloc_3, %alloc_4 : memref<f32> to memref<f32>
    scf.for %arg2 = %c0 to %c5 step %c1 {
      %0 = memref.load %alloc_2[%arg2] : memref<5xf32>
      %1 = memref.load %alloc_4[] : memref<f32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %alloc_4[] : memref<f32>
    }
    gpu.launch_func  @JITOpGenerated0_kernel_0::@JITOpGenerated0_kernel blocks in (%c5, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc_2 : memref<5xf32>, %alloc_4 : memref<f32>, %arg1 : memref<5xf32>)
    return
  }
  gpu.module @JITOpGenerated0_kernel {
    gpu.func @JITOpGenerated0_kernel(%arg0: memref<5xf32>, %arg1: memref<f32>, %arg2: memref<5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = memref.load %arg0[%0] : memref<5xf32>
      %2 = memref.load %arg1[] : memref<f32>
      %3 = arith.subf %1, %2 : f32
      %4 = math.exp %3 : f32
      memref.store %4, %arg2[%0] : memref<5xf32>
      gpu.return
    }
  }
  gpu.module @JITOpGenerated0_kernel_0 {
    gpu.func @JITOpGenerated0_kernel(%arg0: memref<5xf32>, %arg1: memref<f32>, %arg2: memref<5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = memref.load %arg0[%0] : memref<5xf32>
      %2 = memref.load %arg1[] : memref<f32>
      %3 = arith.divf %1, %2 : f32
      memref.store %3, %arg2[%0] : memref<5xf32>
      gpu.return
    }
  }
}

// -----

module attributes {gpu.container_module} {
  func.func @JITOpGenerated0(%arg0: memref<5xf32>, %arg1: memref<5xf32>) attributes {llvm.emit_c_interface, oneflow.cpu} {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant -1.000000e+30 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst_0, %alloc[] : memref<f32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %alloc, %alloc_1 : memref<f32> to memref<f32>
    scf.for %arg2 = %c0 to %c5 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<5xf32>
      %1 = memref.load %alloc_1[] : memref<f32>
      %2 = arith.maxf %0, %1 : f32
      memref.store %2, %alloc_1[] : memref<f32>
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<5xf32>
    gpu.launch_func  @JITOpGenerated0_kernel::@JITOpGenerated0_kernel blocks in (%c5, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg0 : memref<5xf32>, %alloc_1 : memref<f32>, %alloc_2 : memref<5xf32>)
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.store %cst, %alloc_3[] : memref<f32>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %alloc_3, %alloc_4 : memref<f32> to memref<f32>
    scf.for %arg2 = %c0 to %c5 step %c1 {
      %0 = memref.load %alloc_2[%arg2] : memref<5xf32>
      %1 = memref.load %alloc_4[] : memref<f32>
      %2 = arith.addf %0, %1 : f32
      memref.store %2, %alloc_4[] : memref<f32>
    }
    gpu.launch_func  @JITOpGenerated0_kernel_0::@JITOpGenerated0_kernel blocks in (%c5, %c1, %c1) threads in (%c1, %c1, %c1) args(%alloc_2 : memref<5xf32>, %alloc_4 : memref<f32>, %arg1 : memref<5xf32>)
    return
  }
  gpu.module @JITOpGenerated0_kernel {
    gpu.func @JITOpGenerated0_kernel(%arg0: memref<5xf32>, %arg1: memref<f32>, %arg2: memref<5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = memref.load %arg0[%0] : memref<5xf32>
      %2 = memref.load %arg1[] : memref<f32>
      %3 = arith.subf %1, %2 : f32
      %4 = math.exp %3 : f32
      memref.store %4, %arg2[%0] : memref<5xf32>
      gpu.return
    }
  }
  gpu.module @JITOpGenerated0_kernel_0 {
    gpu.func @JITOpGenerated0_kernel(%arg0: memref<5xf32>, %arg1: memref<f32>, %arg2: memref<5xf32>) kernel attributes {gpu.known_block_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = memref.load %arg0[%0] : memref<5xf32>
      %2 = memref.load %arg1[] : memref<f32>
      %3 = arith.divf %1, %2 : f32
      memref.store %3, %arg2[%0] : memref<5xf32>
      gpu.return
    }
  }
}
// RUN: oneflow-opt %s -append-ofstream | FileCheck %s

// CHECK: func.func @JITOpGenerated0(%arg0: memref<1xf32>, %arg1: memref<5xi64>, %arg2: memref<5xf32>, %arg3: !llvm.ptr) attributes {llvm.emit_c_interface}

module attributes {gpu.container_module} {
  func.func @JITOpGenerated0(%arg0: memref<1xf32>, %arg1: memref<5xi64>, %arg2: memref<5xf32>) attributes {llvm.emit_c_interface} {
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %collapse_shape = memref.collapse_shape %arg0 [] : memref<1xf32> into memref<f32>
    gpu.launch_func  @JITOpGenerated0_kernel::@JITOpGenerated0_kernel blocks in (%c5, %c1, %c1) threads in (%c1, %c1, %c1) args(%arg1 : memref<5xi64>, %collapse_shape : memref<f32>, %arg2 : memref<5xf32>)
    return
  }
  gpu.module @JITOpGenerated0_kernel attributes {gpu.binary = ""} {
    llvm.func @JITOpGenerated0_kernel(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<f32>, %arg6: !llvm.ptr<f32>, %arg7: i64, %arg8: !llvm.ptr<f32>, %arg9: !llvm.ptr<f32>, %arg10: i64, %arg11: i64, %arg12: i64) attributes {gpu.kernel, gpu.known_block_size = array<i32: 1, 1, 1>, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
      %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
      %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
      %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
      %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
      %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> 
      %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> 
      %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> 
      %10 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
      %11 = llvm.insertvalue %arg8, %10[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
      %12 = llvm.insertvalue %arg9, %11[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
      %13 = llvm.insertvalue %arg10, %12[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
      %14 = llvm.insertvalue %arg11, %13[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
      %15 = llvm.insertvalue %arg12, %14[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
      %16 = nvvm.read.ptx.sreg.ctaid.x : i32
      %17 = llvm.sext %16 : i32 to i64
      %18 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
      %19 = llvm.getelementptr %18[%17] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
      %20 = llvm.load %19 : !llvm.ptr<i64>
      %21 = llvm.extractvalue %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> 
      %22 = llvm.load %21 : !llvm.ptr<f32>
      %23 = llvm.sitofp %20 : i64 to f32
      %24 = llvm.fmul %23, %22  : f32
      %25 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
      %26 = llvm.getelementptr %25[%17] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
      llvm.store %24, %26 : !llvm.ptr<f32>
      llvm.return
    }
  }
}

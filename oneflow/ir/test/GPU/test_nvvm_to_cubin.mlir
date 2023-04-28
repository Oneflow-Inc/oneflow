// RUN: oneflow-opt %s -pass-pipeline="builtin.module(gpu.module(nvvm-to-cubin))" | FileCheck %s

// CHECK: .text.__nv_logf
// CHECK-SAME: .text.__nv_expf
module attributes {gpu.container_module, oneflow.mempool = 1 : i64} {
  func.func @JITOpGenerated0(%arg0: memref<1xi8>, %arg1: memref<5xi64>, %arg2: memref<1xf32>, %arg3: memref<5xf32>) attributes {llvm.emit_c_interface} {
    return
  }
  gpu.module @JITOpGenerated0_kernel {
    llvm.func @__nv_logf(f32) -> f32
    llvm.func @__nv_expf(f32) -> f32
    llvm.func @JITOpGenerated0(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: !llvm.ptr, %arg13: !llvm.ptr, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: !llvm.ptr, %arg18: !llvm.ptr, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: i64, %arg23: i64, %arg24: !llvm.ptr, %arg25: !llvm.ptr, %arg26: i64, %arg27: i64, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: !llvm.ptr, %arg32: !llvm.ptr, %arg33: i64, %arg34: i64, %arg35: i64, %arg36: i64, %arg37: i64, %arg38: !llvm.ptr, %arg39: !llvm.ptr, %arg40: i64, %arg41: i64, %arg42: i64, %arg43: i64, %arg44: i64, %arg45: !llvm.ptr, %arg46: !llvm.ptr, %arg47: i64, %arg48: i64, %arg49: i64, %arg50: i64, %arg51: i64) attributes {gpu.kernel, nvvm.kernel} {
      %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
      %4 = llvm.insertvalue %arg5, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %5 = llvm.insertvalue %arg6, %4[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %6 = llvm.insertvalue %arg7, %5[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %7 = llvm.insertvalue %arg8, %6[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %8 = llvm.insertvalue %arg12, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %9 = llvm.insertvalue %arg13, %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
      %10 = llvm.insertvalue %arg17, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %11 = llvm.insertvalue %arg18, %10[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %12 = llvm.insertvalue %arg19, %11[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %13 = llvm.insertvalue %arg20, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %14 = llvm.insertvalue %arg24, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %15 = llvm.insertvalue %arg25, %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %16 = llvm.insertvalue %arg26, %15[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %17 = llvm.insertvalue %arg27, %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %18 = llvm.insertvalue %arg31, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %19 = llvm.insertvalue %arg32, %18[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %20 = llvm.insertvalue %arg33, %19[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %21 = llvm.insertvalue %arg34, %20[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %22 = llvm.insertvalue %arg38, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %23 = llvm.insertvalue %arg39, %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %24 = llvm.insertvalue %arg40, %23[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %25 = llvm.insertvalue %arg41, %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %26 = llvm.insertvalue %arg45, %3[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %27 = llvm.insertvalue %arg46, %26[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %28 = llvm.insertvalue %arg47, %27[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %29 = llvm.insertvalue %arg48, %28[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
      %30 = llvm.mlir.constant(0 : index) : i64
      %31 = llvm.mlir.constant(4000 : index) : i64
      %32 = llvm.mlir.constant(1000 : index) : i64
      %33 = llvm.mlir.constant(-1 : index) : i64
      %34 = nvvm.read.ptx.sreg.ctaid.x : i32
      %35 = llvm.sext %34 : i32 to i64
      %36 = nvvm.read.ptx.sreg.ntid.x : i32
      %37 = llvm.sext %36 : i32 to i64
      %38 = nvvm.read.ptx.sreg.tid.x : i32
      %39 = llvm.sext %38 : i32 to i64
      %40 = llvm.mul %37, %35  : i64
      %41 = llvm.add %39, %40  : i64
      %42 = llvm.icmp "slt" %41, %31 : i64
      llvm.cond_br %42, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %43 = llvm.srem %41, %32  : i64
      %44 = llvm.icmp "slt" %43, %30 : i64
      %45 = llvm.add %43, %32  : i64
      %46 = llvm.select %44, %45, %43 : i1, i64
      %47 = llvm.icmp "slt" %41, %30 : i64
      %48 = llvm.sub %33, %41  : i64
      %49 = llvm.select %47, %48, %41 : i1, i64
      %50 = llvm.sdiv %49, %32  : i64
      %51 = llvm.sub %33, %50  : i64
      %52 = llvm.select %47, %51, %50 : i1, i64
      %53 = llvm.mul %52, %32  : i64
      %54 = llvm.add %53, %46  : i64
      %55 = llvm.getelementptr %arg18[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f16
      %56 = llvm.load %55 : !llvm.ptr -> f16
      %57 = llvm.getelementptr %arg6[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f16
      %58 = llvm.load %57 : !llvm.ptr -> f16
      %59 = llvm.getelementptr %arg1[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f16
      %60 = llvm.load %59 : !llvm.ptr -> f16
      %61 = llvm.getelementptr %arg13[%52] : (!llvm.ptr, i64) -> !llvm.ptr, f16
      %62 = llvm.load %61 : !llvm.ptr -> f16
      %63 = llvm.getelementptr %arg25[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %64 = llvm.load %63 : !llvm.ptr -> f32
      %65 = llvm.fpext %60 : f16 to f32
      %66 = llvm.call @__nv_logf(%65) : (f32) -> f32
      %67 = llvm.fptrunc %66 : f32 to f16
      %68 = llvm.fsub %58, %67  : f16
      %69 = llvm.fpext %68 : f16 to f32
      %70 = llvm.call @__nv_expf(%69) : (f32) -> f32
      %71 = llvm.fptrunc %70 : f32 to f16
      %72 = llvm.fmul %71, %62  : f16
      %73 = llvm.fsub %56, %72  : f16
      %74 = llvm.fmul %69, %64  : f32
      %75 = llvm.fpext %73 : f16 to f32
      %76 = llvm.getelementptr %arg32[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f16
      llvm.store %73, %76 : f16, !llvm.ptr
      %77 = llvm.getelementptr %arg39[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %74, %77 : f32, !llvm.ptr
      %78 = llvm.getelementptr %arg46[%54] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      llvm.store %75, %78 : f32, !llvm.ptr
      llvm.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      llvm.return
    }
  }
}
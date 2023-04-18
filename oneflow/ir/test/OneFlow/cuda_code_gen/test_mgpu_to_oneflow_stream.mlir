// RUN: oneflow-opt %s -mgpu-to-ofstream

module attributes {gpu.container_module} {
  llvm.mlir.global internal constant @JITOpGenerated0_kernel_JITOpGenerated0_kernel_kernel_name("JITOpGenerated0_kernel\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @JITOpGenerated0_kernel_gpubin_cst("\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00u\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0A\00\00\00\00\00\00V\05V\00@\00\00\00\00\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.JITOpGenerated0_kernel\00.nv.info.JITOpGenerated0_kernel\00.nv.shared.JITOpGenerated0_kernel\00.nv.constant0.JITOpGenerated0_kernel\00.rel.nv.constant0.JITOpGenerated0_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00JITOpGenerated0_kernel\00.text.JITOpGenerated0_kernel\00.nv.info.JITOpGenerated0_kernel\00.nv.shared.JITOpGenerated0_kernel\00.rel.nv.constant0.JITOpGenerated0_kernel\00.nv.constant0.JITOpGenerated0_kernel\00_param\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00I\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D1\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\FD\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00-\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00I\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\FF\FF\FF\FF(\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\00\00\00\00\FF\FF\FF\FF0\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F0\01\00\00\00\00\00\00\04\04\00\00\00\04<\00\00\00\0C\81\80\80(\00\04\FC\FF\FF?\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0E\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\1C\04\00\F0\00\00\00\03\1B\FF\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\04\17\0C\00\00\00\00\00\07\008\00\00\F0!\00\04\17\0C\00\00\00\00\00\08\00@\00\00\F0!\00\04\17\0C\00\00\00\00\00\09\00H\00\00\F0!\00\04\17\0C\00\00\00\00\00\0A\00P\00\00\F0!\00\04\17\0C\00\00\00\00\00\0B\00X\00\00\F0!\00\04\17\0C\00\00\00\00\00\0C\00`\00\00\F0!\00\03\19h\00\04\0A\08\00\02\00\00\00`\01h\00\015\00\00\047\04\00u\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00K\00\00\00\00\00\00\00\00\02\02\08\10\0A/\22\00\00\00\08\00\00\00\00\00\00\08\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\18\08\00\00\00\00\00\00 \08\00\00\00\00\00\00(\08\00\00\00\00\00\000\08\00\00\00\00\00\008\08\00\00\00\00\01\00\00\08\00\00\00\00\01\00\08\08\00\00\00\00\01\00\10\08\00\00\00\00\01\00\18\08\00\00\00\00\01\00 \08\00\00\00\00\01\00(\08\00\00\00\00\01\000\08\00\00\00\00\01\008\08\00\00\00\00\02\00\00\08\00\00\00\00\02\00\08\08\00\00\00\00\02\00\10\08\00\00\00\00\02\00\18\08\00\00\00\00\02\00 \08\00\00\00\00\02\00(\08\00\00\00\00\02\000\08\00\00\00\00\02\008\08\00\00\00\00\00\00\00\14,\00\00\00\09\00\00\0C\00\00\00\00H\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02z\01\00\00\0A\00\00\00\0F\00\00\00\C4\0F\00\19y\06\00\00\00\00\00\00%\00\00\00\22\0E\00\02x\03\00\08\00\00\00\00\0F\00\00\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\E2\0F\00\02z\04\00\00d\00\00\00\0F\00\00\00\E4\0F\00\02z\05\00\00e\00\00\00\0F\00\00\00\CA\0F\00\80y\04\04\04\00\00\00\00\19\10\0C\00\A2\0E\00%v\02\06\00Z\00\00\03\02\8E\07\00\CA\1F\00\80y\08\02\04\00\00\00\00\19\10\0C\00\E8\0E\00\80y\09\02\04\04\00\00\00\19\10\0C\00\E2\0E\00\02x\07\00\04\00\00\00\00\0F\00\00\00\CA\0F\00%v\06\06\00j\00\00\07\02\8E\07\00\E2\0F\00\12s\09\00\08\00\00\00\00\140\00\00\A4\8E\00 r\0B\04\09\00\00\00\00\00@\00\00\CAO\00\85y\00\06\0B\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00X\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D8\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F0\03\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\14\04\00\00\00\00\00\00\F8\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0C\05\00\00\00\00\00\00\18\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\05\00\00\00\00\00\00\E0\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\06\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\06\00\00\00\00\00\00\C8\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0E\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00") {addr_space = 0 : i32}
  llvm.func @JITOpGenerated0(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr<i64>, %arg6: !llvm.ptr<i64>, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr<f32>, %arg11: !llvm.ptr<f32>, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = builtin.unrealized_conversion_cast %5 : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> to memref<1xf32>
    %7 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
    %8 = llvm.insertvalue %arg5, %7[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg6, %8[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg7, %9[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg8, %10[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %arg9, %11[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %14 = llvm.insertvalue %arg10, %13[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %arg11, %14[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg12, %15[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg13, %16[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.insertvalue %arg14, %17[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.mlir.constant(5 : index) : i64
    %20 = llvm.mlir.constant(1 : index) : i64
    %collapse_shape = memref.collapse_shape %6 [] : memref<1xf32> into memref<f32>
    %21 = builtin.unrealized_conversion_cast %collapse_shape : memref<f32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64)>
    %22 = llvm.mlir.addressof @JITOpGenerated0_kernel_gpubin_cst : !llvm.ptr<array<3328 x i8>>
    %23 = llvm.getelementptr %22[0, 0] : (!llvm.ptr<array<3328 x i8>>) -> !llvm.ptr<i8>
    %24 = llvm.call @mgpuModuleLoad(%23) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %25 = llvm.mlir.addressof @JITOpGenerated0_kernel_JITOpGenerated0_kernel_kernel_name : !llvm.ptr<array<23 x i8>>
    %26 = llvm.getelementptr %25[0, 0] : (!llvm.ptr<array<23 x i8>>) -> !llvm.ptr<i8>
    %27 = llvm.call @mgpuModuleGetFunction(%24, %26) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    %28 = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NOT: mgpuStreamCreate
    %29 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr<i8>
    %30 = llvm.extractvalue %12[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %31 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.extractvalue %21[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> 
    %36 = llvm.extractvalue %21[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> 
    %37 = llvm.extractvalue %21[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> 
    %38 = llvm.extractvalue %18[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.extractvalue %18[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.extractvalue %18[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %42 = llvm.extractvalue %18[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.mlir.constant(1 : i32) : i32
    %44 = llvm.alloca %43 x !llvm.struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)> : (i32) -> !llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>
    %45 = llvm.mlir.constant(13 : i32) : i32
    %46 = llvm.alloca %45 x !llvm.ptr<i8> : (i32) -> !llvm.ptr<ptr<i8>>
    %47 = llvm.getelementptr %44[0, 0] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<i64>>
    llvm.store %30, %47 : !llvm.ptr<ptr<i64>>
    %48 = llvm.getelementptr %46[0] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %49 = llvm.bitcast %47 : !llvm.ptr<ptr<i64>> to !llvm.ptr<i8>
    llvm.store %49, %48 : !llvm.ptr<ptr<i8>>
    %50 = llvm.getelementptr %44[0, 1] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<i64>>
    llvm.store %31, %50 : !llvm.ptr<ptr<i64>>
    %51 = llvm.getelementptr %46[1] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %52 = llvm.bitcast %50 : !llvm.ptr<ptr<i64>> to !llvm.ptr<i8>
    llvm.store %52, %51 : !llvm.ptr<ptr<i8>>
    %53 = llvm.getelementptr %44[0, 2] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %32, %53 : !llvm.ptr<i64>
    %54 = llvm.getelementptr %46[2] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %55 = llvm.bitcast %53 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %55, %54 : !llvm.ptr<ptr<i8>>
    %56 = llvm.getelementptr %44[0, 3] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %33, %56 : !llvm.ptr<i64>
    %57 = llvm.getelementptr %46[3] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %58 = llvm.bitcast %56 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %58, %57 : !llvm.ptr<ptr<i8>>
    %59 = llvm.getelementptr %44[0, 4] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %34, %59 : !llvm.ptr<i64>
    %60 = llvm.getelementptr %46[4] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %61 = llvm.bitcast %59 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %61, %60 : !llvm.ptr<ptr<i8>>
    %62 = llvm.getelementptr %44[0, 5] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<f32>>
    llvm.store %35, %62 : !llvm.ptr<ptr<f32>>
    %63 = llvm.getelementptr %46[5] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %64 = llvm.bitcast %62 : !llvm.ptr<ptr<f32>> to !llvm.ptr<i8>
    llvm.store %64, %63 : !llvm.ptr<ptr<i8>>
    %65 = llvm.getelementptr %44[0, 6] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<f32>>
    llvm.store %36, %65 : !llvm.ptr<ptr<f32>>
    %66 = llvm.getelementptr %46[6] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %67 = llvm.bitcast %65 : !llvm.ptr<ptr<f32>> to !llvm.ptr<i8>
    llvm.store %67, %66 : !llvm.ptr<ptr<i8>>
    %68 = llvm.getelementptr %44[0, 7] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %37, %68 : !llvm.ptr<i64>
    %69 = llvm.getelementptr %46[7] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %70 = llvm.bitcast %68 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %70, %69 : !llvm.ptr<ptr<i8>>
    %71 = llvm.getelementptr %44[0, 8] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<f32>>
    llvm.store %38, %71 : !llvm.ptr<ptr<f32>>
    %72 = llvm.getelementptr %46[8] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %73 = llvm.bitcast %71 : !llvm.ptr<ptr<f32>> to !llvm.ptr<i8>
    llvm.store %73, %72 : !llvm.ptr<ptr<i8>>
    %74 = llvm.getelementptr %44[0, 9] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<f32>>
    llvm.store %39, %74 : !llvm.ptr<ptr<f32>>
    %75 = llvm.getelementptr %46[9] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %76 = llvm.bitcast %74 : !llvm.ptr<ptr<f32>> to !llvm.ptr<i8>
    llvm.store %76, %75 : !llvm.ptr<ptr<i8>>
    %77 = llvm.getelementptr %44[0, 10] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %40, %77 : !llvm.ptr<i64>
    %78 = llvm.getelementptr %46[10] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %79 = llvm.bitcast %77 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %79, %78 : !llvm.ptr<ptr<i8>>
    %80 = llvm.getelementptr %44[0, 11] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %41, %80 : !llvm.ptr<i64>
    %81 = llvm.getelementptr %46[11] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %82 = llvm.bitcast %80 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %82, %81 : !llvm.ptr<ptr<i8>>
    %83 = llvm.getelementptr %44[0, 12] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %42, %83 : !llvm.ptr<i64>
    %84 = llvm.getelementptr %46[12] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %85 = llvm.bitcast %83 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %85, %84 : !llvm.ptr<ptr<i8>>
    %86 = llvm.mlir.null : !llvm.ptr<ptr<i8>>
    // CHECK-NOT: mgpuLaunchKernel(%18, %4, %3, %3, %3, %3, %3, %2, %arg15, %23, %62)
    llvm.call @mgpuLaunchKernel(%27, %19, %20, %20, %20, %20, %20, %28, %29, %46, %86) : (!llvm.ptr<i8>, i64, i64, i64, i64, i64, i64, i32, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>, !llvm.ptr<ptr<i8>>) -> ()
    llvm.call @mgpuStreamSynchronize(%29) : (!llvm.ptr<i8>) -> ()
    // CHECK-NOT: mgpuStreamDestroy
    llvm.call @mgpuStreamDestroy(%29) : (!llvm.ptr<i8>) -> ()
    llvm.call @mgpuModuleUnload(%24) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_JITOpGenerated0(%arg0: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, %arg1: !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>, %arg2: !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>, %arg3: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg0 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.load %arg2 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %13 = llvm.extractvalue %12[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @JITOpGenerated0(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %13, %14, %15, %16, %17, %arg3) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @mgpuModuleLoad(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  llvm.func @mgpuModuleGetFunction(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
  llvm.func @mgpuStreamCreate() -> !llvm.ptr<i8>
  llvm.func @mgpuLaunchKernel(!llvm.ptr<i8>, i64, i64, i64, i64, i64, i64, i32, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>, !llvm.ptr<ptr<i8>>)
  llvm.func @mgpuStreamSynchronize(!llvm.ptr<i8>)
  llvm.func @mgpuStreamDestroy(!llvm.ptr<i8>)
  llvm.func @mgpuModuleUnload(!llvm.ptr<i8>)
}
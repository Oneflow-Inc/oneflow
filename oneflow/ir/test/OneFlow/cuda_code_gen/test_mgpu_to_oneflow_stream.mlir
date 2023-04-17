// RUN: oneflow-opt %s -mgpu-to-ofstream

module attributes {gpu.container_module} {
  llvm.mlir.global internal constant @JITOpGenerated0_kernel_JITOpGenerated0_kernel_kernel_name("JITOpGenerated0_kernel\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @JITOpGenerated0_kernel_gpubin_cst("\7FELF\02\01\013\07\00\00\00\00\00\00\00\02\00\BE\00u\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0A\00\00\00\00\00\00V\05V\00@\00\00\00\00\00@\00\0C\00\01\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00.text.JITOpGenerated0_kernel\00.nv.info.JITOpGenerated0_kernel\00.nv.shared.JITOpGenerated0_kernel\00.nv.constant0.JITOpGenerated0_kernel\00.rel.nv.constant0.JITOpGenerated0_kernel\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00.shstrtab\00.strtab\00.symtab\00.symtab_shndx\00.nv.info\00JITOpGenerated0_kernel\00.text.JITOpGenerated0_kernel\00.nv.info.JITOpGenerated0_kernel\00.nv.shared.JITOpGenerated0_kernel\00.rel.nv.constant0.JITOpGenerated0_kernel\00.nv.constant0.JITOpGenerated0_kernel\00_param\00.debug_frame\00.rel.debug_frame\00.rela.debug_frame\00.nv.callgraph\00.nv.prototype\00.nv.rel.action\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00I\00\00\00\03\00\0B\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D1\00\00\00\03\00\0A\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\FD\00\00\00\03\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00-\01\00\00\03\00\07\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00I\01\00\00\03\00\08\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\12\10\0B\00\00\00\00\00\00\00\00\00\00\02\00\00\00\00\00\00\FF\FF\FF\FF(\00\00\00\00\00\00\00\FF\FF\FF\FF\FF\FF\FF\FF\03\00\04|\FF\FF\FF\FF\0F\0C\81\80\80(\00\08\FF\81\80(\08\81\80\80(\00\00\00\00\00\00\00\FF\FF\FF\FF0\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F0\01\00\00\00\00\00\00\04\04\00\00\00\04<\00\00\00\0C\81\80\80(\00\04\FC\FF\FF?\00\00\00\04\11\08\00\06\00\00\00\00\00\00\00\04/\08\00\06\00\00\00\0E\00\00\00\04\12\08\00\06\00\00\00\00\00\00\00\04\1C\04\00\F0\00\00\00\03\1B\FF\00\04\17\0C\00\00\00\00\00\00\00\00\00\00\F0!\00\04\17\0C\00\00\00\00\00\01\00\08\00\00\F0!\00\04\17\0C\00\00\00\00\00\02\00\10\00\00\F0!\00\04\17\0C\00\00\00\00\00\03\00\18\00\00\F0!\00\04\17\0C\00\00\00\00\00\04\00 \00\00\F0!\00\04\17\0C\00\00\00\00\00\05\00(\00\00\F0!\00\04\17\0C\00\00\00\00\00\06\000\00\00\F0!\00\04\17\0C\00\00\00\00\00\07\008\00\00\F0!\00\04\17\0C\00\00\00\00\00\08\00@\00\00\F0!\00\04\17\0C\00\00\00\00\00\09\00H\00\00\F0!\00\04\17\0C\00\00\00\00\00\0A\00P\00\00\F0!\00\04\17\0C\00\00\00\00\00\0B\00X\00\00\F0!\00\04\17\0C\00\00\00\00\00\0C\00`\00\00\F0!\00\03\19h\00\04\0A\08\00\02\00\00\00`\01h\00\015\00\00\047\04\00u\00\00\00\00\00\00\00\FF\FF\FF\FF\00\00\00\00\FE\FF\FF\FF\00\00\00\00\FD\FF\FF\FF\00\00\00\00K\00\00\00\00\00\00\00\00\02\02\08\10\0A/\22\00\00\00\08\00\00\00\00\00\00\08\08\00\00\00\00\00\00\10\08\00\00\00\00\00\00\18\08\00\00\00\00\00\00 \08\00\00\00\00\00\00(\08\00\00\00\00\00\000\08\00\00\00\00\00\008\08\00\00\00\00\01\00\00\08\00\00\00\00\01\00\08\08\00\00\00\00\01\00\10\08\00\00\00\00\01\00\18\08\00\00\00\00\01\00 \08\00\00\00\00\01\00(\08\00\00\00\00\01\000\08\00\00\00\00\01\008\08\00\00\00\00\02\00\00\08\00\00\00\00\02\00\08\08\00\00\00\00\02\00\10\08\00\00\00\00\02\00\18\08\00\00\00\00\02\00 \08\00\00\00\00\02\00(\08\00\00\00\00\02\000\08\00\00\00\00\02\008\08\00\00\00\00\00\00\00\14,\00\00\00\09\00\00\0C\00\00\00\00H\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\02z\01\00\00\0A\00\00\00\0F\00\00\00\C4\0F\00\19y\06\00\00\00\00\00\00%\00\00\00\22\0E\00\02x\03\00\08\00\00\00\00\0F\00\00\00\E2\0F\00\B9z\04\00\00F\00\00\00\0A\00\00\00\E2\0F\00\02z\04\00\00d\00\00\00\0F\00\00\00\E4\0F\00\02z\05\00\00e\00\00\00\0F\00\00\00\CA\0F\00\80y\04\04\04\00\00\00\00\19\10\0C\00\A2\0E\00%v\02\06\00Z\00\00\03\02\8E\07\00\CA\1F\00\80y\08\02\04\00\00\00\00\19\10\0C\00\E8\0E\00\80y\09\02\04\04\00\00\00\19\10\0C\00\E2\0E\00\02x\07\00\04\00\00\00\00\0F\00\00\00\CA\0F\00%v\06\06\00j\00\00\07\02\8E\07\00\E2\0F\00\12s\09\00\08\00\00\00\00\140\00\00\A4\8E\00 r\0B\04\09\00\00\00\00\00@\00\00\CAO\00\85y\00\06\0B\00\00\00\04\19\10\0C\00\E2\0F\00My\00\00\00\00\00\00\00\00\80\03\00\EA\0F\00Gy\00\00\F0\FF\FF\FF\FF\FF\83\03\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\18y\00\00\00\00\00\00\00\00\00\00\00\C0\0F\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00@\00\00\00\00\00\00\00:\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\03\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00z\01\00\00\00\00\00\00X\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\13\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\D8\02\00\00\00\00\00\00\A8\00\00\00\00\00\00\00\02\00\00\00\06\00\00\00\08\00\00\00\00\00\00\00\18\00\00\00\00\00\00\00\DF\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\80\03\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\01\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00)\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\F0\03\00\00\00\00\00\00$\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00O\00\00\00\00\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\14\04\00\00\00\00\00\00\F8\00\00\00\00\00\00\00\03\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0F\01\00\00\01\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\0C\05\00\00\00\00\00\00\18\00\00\00\00\00\00\00\03\00\00\00\00\00\00\00\04\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00+\01\00\00\0B\00\00p\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00(\05\00\00\00\00\00\00\E0\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\EC\00\00\00\09\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\06\00\00\00\00\00\00\10\00\00\00\00\00\00\00\03\00\00\00\04\00\00\00\08\00\00\00\00\00\00\00\10\00\00\00\00\00\00\00\91\00\00\00\01\00\00\00\02\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\18\06\00\00\00\00\00\00\C8\01\00\00\00\00\00\00\00\00\00\00\0B\00\00\00\04\00\00\00\00\00\00\00\00\00\00\00\00\00\00\002\00\00\00\01\00\00\00\06\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\08\00\00\00\00\00\00\00\02\00\00\00\00\00\00\03\00\00\00\06\00\00\0E\80\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00") {addr_space = 0 : i32}
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
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
    %22 = llvm.mlir.constant(5 : index) : i64
    %23 = llvm.mlir.constant(1 : index) : i64
    %24 = llvm.mlir.null : !llvm.ptr<f32>
    %25 = llvm.getelementptr %24[%22] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %26 = llvm.ptrtoint %25 : !llvm.ptr<f32> to i64
    %27 = llvm.mlir.constant(64 : index) : i64
    %28 = llvm.add %26, %27  : i64
    %29 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr<i8>
    %30 = llvm.bitcast %29 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %31 = llvm.ptrtoint %30 : !llvm.ptr<f32> to i64
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.sub %27, %32  : i64
    %34 = llvm.add %31, %33  : i64
    %35 = llvm.urem %34, %27  : i64
    %36 = llvm.sub %34, %35  : i64
    %37 = llvm.inttoptr %36 : i64 to !llvm.ptr<f32>
    %38 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %39 = llvm.insertvalue %30, %38[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %40 = llvm.insertvalue %37, %39[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.mlir.constant(0 : index) : i64
    %42 = llvm.insertvalue %41, %40[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.insertvalue %22, %42[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %44 = llvm.insertvalue %23, %43[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.alloca %45 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    llvm.store %44, %46 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>>
    %47 = llvm.bitcast %46 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
    %48 = llvm.mlir.constant(1 : index) : i64
    %49 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %50 = llvm.insertvalue %48, %49[0] : !llvm.struct<(i64, ptr<i8>)> 
    %51 = llvm.insertvalue %47, %50[1] : !llvm.struct<(i64, ptr<i8>)> 
    %52 = llvm.mlir.null : !llvm.ptr<f32>
    %53 = llvm.getelementptr %52[1] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %54 = llvm.ptrtoint %53 : !llvm.ptr<f32> to i64
    %55 = llvm.extractvalue %51[0] : !llvm.struct<(i64, ptr<i8>)> 
    %56 = llvm.extractvalue %51[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @mgpuMemHostRegisterMemRef(%55, %56, %54) : (i64, !llvm.ptr<i8>, i64) -> ()
    %57 = llvm.mlir.addressof @JITOpGenerated0_kernel_gpubin_cst : !llvm.ptr<array<3328 x i8>>
    %58 = llvm.getelementptr %57[0, 0] : (!llvm.ptr<array<3328 x i8>>) -> !llvm.ptr<i8>
    %59 = llvm.call @mgpuModuleLoad(%58) : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
    %60 = llvm.mlir.addressof @JITOpGenerated0_kernel_JITOpGenerated0_kernel_kernel_name : !llvm.ptr<array<23 x i8>>
    %61 = llvm.getelementptr %60[0, 0] : (!llvm.ptr<array<23 x i8>>) -> !llvm.ptr<i8>
    %62 = llvm.call @mgpuModuleGetFunction(%59, %61) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
    %63 = llvm.mlir.constant(0 : i32) : i32
    %64 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr<i8>
    %65 = llvm.extractvalue %12[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %66 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %69 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
    %70 = llvm.extractvalue %21[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> 
    %71 = llvm.extractvalue %21[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> 
    %72 = llvm.extractvalue %21[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64)> 
    %73 = llvm.extractvalue %44[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.extractvalue %44[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %75 = llvm.extractvalue %44[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %76 = llvm.extractvalue %44[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.extractvalue %44[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %78 = llvm.mlir.constant(1 : i32) : i32
    %79 = llvm.alloca %78 x !llvm.struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)> : (i32) -> !llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>
    %80 = llvm.mlir.constant(13 : i32) : i32
    %81 = llvm.alloca %80 x !llvm.ptr<i8> : (i32) -> !llvm.ptr<ptr<i8>>
    %82 = llvm.getelementptr %79[0, 0] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<i64>>
    llvm.store %65, %82 : !llvm.ptr<ptr<i64>>
    %83 = llvm.getelementptr %81[0] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %84 = llvm.bitcast %82 : !llvm.ptr<ptr<i64>> to !llvm.ptr<i8>
    llvm.store %84, %83 : !llvm.ptr<ptr<i8>>
    %85 = llvm.getelementptr %79[0, 1] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<i64>>
    llvm.store %66, %85 : !llvm.ptr<ptr<i64>>
    %86 = llvm.getelementptr %81[1] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %87 = llvm.bitcast %85 : !llvm.ptr<ptr<i64>> to !llvm.ptr<i8>
    llvm.store %87, %86 : !llvm.ptr<ptr<i8>>
    %88 = llvm.getelementptr %79[0, 2] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %67, %88 : !llvm.ptr<i64>
    %89 = llvm.getelementptr %81[2] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %90 = llvm.bitcast %88 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %90, %89 : !llvm.ptr<ptr<i8>>
    %91 = llvm.getelementptr %79[0, 3] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %68, %91 : !llvm.ptr<i64>
    %92 = llvm.getelementptr %81[3] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %93 = llvm.bitcast %91 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %93, %92 : !llvm.ptr<ptr<i8>>
    %94 = llvm.getelementptr %79[0, 4] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %69, %94 : !llvm.ptr<i64>
    %95 = llvm.getelementptr %81[4] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %96 = llvm.bitcast %94 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %96, %95 : !llvm.ptr<ptr<i8>>
    %97 = llvm.getelementptr %79[0, 5] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<f32>>
    llvm.store %70, %97 : !llvm.ptr<ptr<f32>>
    %98 = llvm.getelementptr %81[5] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %99 = llvm.bitcast %97 : !llvm.ptr<ptr<f32>> to !llvm.ptr<i8>
    llvm.store %99, %98 : !llvm.ptr<ptr<i8>>
    %100 = llvm.getelementptr %79[0, 6] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<f32>>
    llvm.store %71, %100 : !llvm.ptr<ptr<f32>>
    %101 = llvm.getelementptr %81[6] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %102 = llvm.bitcast %100 : !llvm.ptr<ptr<f32>> to !llvm.ptr<i8>
    llvm.store %102, %101 : !llvm.ptr<ptr<i8>>
    %103 = llvm.getelementptr %79[0, 7] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %72, %103 : !llvm.ptr<i64>
    %104 = llvm.getelementptr %81[7] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %105 = llvm.bitcast %103 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %105, %104 : !llvm.ptr<ptr<i8>>
    %106 = llvm.getelementptr %79[0, 8] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<f32>>
    llvm.store %73, %106 : !llvm.ptr<ptr<f32>>
    %107 = llvm.getelementptr %81[8] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %108 = llvm.bitcast %106 : !llvm.ptr<ptr<f32>> to !llvm.ptr<i8>
    llvm.store %108, %107 : !llvm.ptr<ptr<i8>>
    %109 = llvm.getelementptr %79[0, 9] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<ptr<f32>>
    llvm.store %74, %109 : !llvm.ptr<ptr<f32>>
    %110 = llvm.getelementptr %81[9] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %111 = llvm.bitcast %109 : !llvm.ptr<ptr<f32>> to !llvm.ptr<i8>
    llvm.store %111, %110 : !llvm.ptr<ptr<i8>>
    %112 = llvm.getelementptr %79[0, 10] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %75, %112 : !llvm.ptr<i64>
    %113 = llvm.getelementptr %81[10] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %114 = llvm.bitcast %112 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %114, %113 : !llvm.ptr<ptr<i8>>
    %115 = llvm.getelementptr %79[0, 11] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %76, %115 : !llvm.ptr<i64>
    %116 = llvm.getelementptr %81[11] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %117 = llvm.bitcast %115 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %117, %116 : !llvm.ptr<ptr<i8>>
    %118 = llvm.getelementptr %79[0, 12] : (!llvm.ptr<struct<"", (ptr<i64>, ptr<i64>, i64, i64, i64, ptr<f32>, ptr<f32>, i64, ptr<f32>, ptr<f32>, i64, i64, i64)>>) -> !llvm.ptr<i64>
    llvm.store %77, %118 : !llvm.ptr<i64>
    %119 = llvm.getelementptr %81[12] : (!llvm.ptr<ptr<i8>>) -> !llvm.ptr<ptr<i8>>
    %120 = llvm.bitcast %118 : !llvm.ptr<i64> to !llvm.ptr<i8>
    llvm.store %120, %119 : !llvm.ptr<ptr<i8>>
    %121 = llvm.mlir.null : !llvm.ptr<ptr<i8>>
    llvm.call @mgpuLaunchKernel(%62, %19, %20, %20, %20, %20, %20, %63, %64, %81, %121) : (!llvm.ptr<i8>, i64, i64, i64, i64, i64, i64, i32, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>, !llvm.ptr<ptr<i8>>) -> ()
    llvm.call @mgpuStreamSynchronize(%64) : (!llvm.ptr<i8>) -> ()
    llvm.call @mgpuStreamDestroy(%64) : (!llvm.ptr<i8>) -> ()
    llvm.call @mgpuModuleUnload(%59) : (!llvm.ptr<i8>) -> ()
    %122 = llvm.call @mgpuStreamCreate() : () -> !llvm.ptr<i8>
    %123 = llvm.mlir.constant(5 : index) : i64
    %124 = llvm.mlir.null : !llvm.ptr<f32>
    %125 = llvm.getelementptr %124[%123] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %126 = llvm.ptrtoint %125 : !llvm.ptr<f32> to i64
    %127 = llvm.extractvalue %44[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %128 = llvm.bitcast %127 : !llvm.ptr<f32> to !llvm.ptr<i8>
    %129 = llvm.extractvalue %18[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %130 = llvm.bitcast %129 : !llvm.ptr<f32> to !llvm.ptr<i8>
    llvm.call @mgpuMemcpy(%130, %128, %126, %122) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) -> ()
    llvm.call @mgpuStreamSynchronize(%122) : (!llvm.ptr<i8>) -> ()
    llvm.call @mgpuStreamDestroy(%122) : (!llvm.ptr<i8>) -> ()
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
  llvm.func @mgpuMemHostRegisterMemRef(i64, !llvm.ptr<i8>, i64)
  llvm.func @mgpuModuleLoad(!llvm.ptr<i8>) -> !llvm.ptr<i8>
  llvm.func @mgpuModuleGetFunction(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
  llvm.func @mgpuStreamCreate() -> !llvm.ptr<i8>
  llvm.func @mgpuLaunchKernel(!llvm.ptr<i8>, i64, i64, i64, i64, i64, i64, i32, !llvm.ptr<i8>, !llvm.ptr<ptr<i8>>, !llvm.ptr<ptr<i8>>)
  llvm.func @mgpuStreamSynchronize(!llvm.ptr<i8>)
  llvm.func @mgpuStreamDestroy(!llvm.ptr<i8>)
  llvm.func @mgpuModuleUnload(!llvm.ptr<i8>)
  llvm.func @mgpuMemcpy(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, !llvm.ptr<i8>)
}
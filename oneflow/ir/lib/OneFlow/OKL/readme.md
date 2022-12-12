## okl dialect
设置 ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH = 1，启动oneflow kernel launch功能打包计算型op。

在roundtrip的结尾，用于计算的连续的op被合并成单个kernel launch op
oneflow ops -> oneflow.kernel_launch{mlir_assembly="wrap"}
例如
```mlir
module {
  oneflow.job @GraphToRun_1(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_input.0.0_2", output_lbns = ["_GraphToRun_1_input.0.0_2/out"], scope_symbol_id = 30 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    %0 = "oneflow.relu"(%output) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %output_0 = "oneflow.output"(%1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_output.0.0_2", output_lbns = ["_GraphToRun_1_output.0.0_2/out"], scope_symbol_id = 30 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    oneflow.return %output_0 : tensor<2xf32>
  }
}
```
中的连续的relu和tanh两个op被合并成如下单个的kernel launch op，其资源在kernel launch op的wrap 函数中以一定的规则得以映射
```mlir
module {
  func.func @wrap0(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
  oneflow.job @GraphToRun_1(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_input.0.0_2", output_lbns = ["_GraphToRun_1_input.0.0_2/out"], scope_symbol_id = 30 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    %0:2 = oneflow.kernel_launch @wrap0(%output) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], mlir_assembly = "\22func.func\22() ({\0A^bb0(%arg0: tensor<2xf32>):\0A  %0 = \22oneflow.relu\22(%arg0) {device_name = [\22@0:0\22], device_tag = \22cuda\22, hierarchy = [1], op_name = \22relu-0\22, scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>\0A  %1 = \22oneflow.tanh\22(%0) {device_name = [\22@0:0\22], device_tag = \22cuda\22, hierarchy = [1], op_name = \22tanh-1\22, scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>\0A  \22func.return\22(%0, %1) : (tensor<2xf32>, tensor<2xf32>) -> ()\0A}) {function_type = (tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>), llvm.emit_c_interface, sym_name = \22wrap0\22} : () -> ()", op_name = "wrap0", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    %output_0 = "oneflow.output"(%0#1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_output.0.0_2", output_lbns = ["_GraphToRun_1_output.0.0_2/out"], scope_symbol_id = 30 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    oneflow.return %output_0 : tensor<2xf32>
  }
}
```

在kernel launch的类中，通过wrap函数的func type来推导input和output形状
 - set input shape(infer from function_type operands)
 - set output shape(infer from function_type results)

在kernel launch的初始化过程中
 - lower mlir_assembly="wrap" to okl module
 - globally create launcher context(init_context, okl_module) once
   - vec<reg_ctx> : datatype/device/input/output(information from wrap op)
   - vec<run_ctx> : input/output for kernel compute function(information from reg_ctx, tensor from init_context)
   - vec<kernel> : datatype/device(information from reg_ctx)
 - lower okl compute to llvm module

其中 lower oneflow to okl 将：
```mlir
module {
  func.func @wrap0(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
```
通过如下几个pass转换成okl格式：
 - -extract-kernel-launch-tensor
 - -trim-return-to-void
 - -lower-to-okl
 - -split-into-funcs
 - -fetch-from-launcher
结果为：
```mlir
module {
  func.func @okl_recycle(%arg0: !okl.launcher_ctx) {
    %0 = "okl.fetch_reg_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.reg_ctx
    %1 = "okl.fetch_reg_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.reg_ctx
    %2 = "okl.fetch_run_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %3 = "okl.fetch_run_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    "okl.destroy_reg_ctx"(%0) : (!okl.reg_ctx) -> ()
    "okl.destroy_reg_ctx"(%1) : (!okl.reg_ctx) -> ()
    "okl.destroy_run_ctx"(%2) : (!okl.run_ctx) -> ()
    "okl.destroy_run_ctx"(%3) : (!okl.run_ctx) -> ()
    return
  }
  func.func @okl_compute(%arg0: !okl.launcher_ctx) {
    %0 = "okl.fetch_run_ctx"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %1 = "okl.fetch_run_ctx"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.run_ctx
    %2 = "okl.fetch_kernel"(%arg0) {index = 0 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    %3 = "okl.fetch_kernel"(%arg0) {index = 1 : si64} : (!okl.launcher_ctx) -> !okl.kernel
    "okl.launch"(%0, %2) : (!okl.run_ctx, !okl.kernel) -> ()
    "okl.launch"(%1, %3) : (!okl.run_ctx, !okl.kernel) -> ()
    return
  }
  func.func @okl_init_context(%arg0: !okl.launcher_ctx) {
    %0 = "okl.build_reg_ctx"() ({
      %6 = "okl.get_tensor_from_arg"(%arg0) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %7 = "oneflow.relu"(%6) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %8 = "okl.get_tensor_as_ret"(%arg0, %7) {index = 0 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {function_type = () -> ()} : () -> !okl.reg_ctx
    %1 = "okl.build_reg_ctx"() ({
      %6 = "okl.get_tensor_from_ret"(%arg0) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %7 = "oneflow.tanh"(%6) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %8 = "okl.get_tensor_as_ret"(%arg0, %7) {index = 1 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {function_type = () -> ()} : () -> !okl.reg_ctx
    %2 = "okl.build_run_ctx"(%0) : (!okl.reg_ctx) -> !okl.run_ctx
    %3 = "okl.build_run_ctx"(%1) : (!okl.reg_ctx) -> !okl.run_ctx
    %4 = "okl.build_op_kernel"(%0) : (!okl.reg_ctx) -> !okl.kernel
    %5 = "okl.build_op_kernel"(%1) : (!okl.reg_ctx) -> !okl.kernel
    return
  }
  func.func private @get_resources_type_0(!okl.launcher_ctx) -> (!okl.reg_ctx, !okl.reg_ctx)
  func.func private @get_resources_type_1(!okl.launcher_ctx) -> (!okl.run_ctx, !okl.run_ctx)
  func.func private @get_resources_type_2(!okl.launcher_ctx) -> (!okl.kernel, !okl.kernel)
}
```
其中okl_init_context用来实现第二部的资源初始化，okl_compute用来后面调用各自的kernel运行compute。

通过如下几个pass，将okl转换成llvm格式：
 - -only-keep-compute-ops
 - -lower-launcher-to-llvm-ptr
 - -lower-okl-to-llvm-call
 - -reconcile-unrealized-casts
 - -convert-func-to-llvm
```mlir
module attributes {llvm.data_layout = ""} {
  llvm.func @launch(!llvm.ptr<i8>, !llvm.ptr<i8>) attributes {llvm.emit_c_interface}
  llvm.func @fetch_kernel(!llvm.ptr<i8>, i64) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  llvm.func @fetch_run_ctx(!llvm.ptr<i8>, i64) -> !llvm.ptr<i8> attributes {llvm.emit_c_interface}
  llvm.func @okl_compute(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.call @fetch_run_ctx(%arg0, %1) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.call @fetch_run_ctx(%arg0, %3) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %5 = llvm.mlir.constant(0 : index) : i64
    %6 = llvm.call @fetch_kernel(%arg0, %5) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    %7 = llvm.mlir.constant(1 : index) : i64
    %8 = llvm.call @fetch_kernel(%arg0, %7) : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
    llvm.call @launch(%2, %6) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    llvm.call @launch(%4, %8) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_okl_compute(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    llvm.call @okl_compute(%arg0) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
}
```
最后compute函数中通过llvm engine运行该llvm
Compute[may execute many times]
 - llvm.engine(llvm module, launcher context)
   - fetch run_ctx(from launcher context with index)
   - fetch kernel(from launcher context with index)
   - launch kernel(run_ctx)

llvm.call的callee逻辑在liboneflow.so，通过extern C实现

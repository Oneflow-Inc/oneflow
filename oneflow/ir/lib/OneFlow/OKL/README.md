# okl dialect

oneflow kernel launch dialect

将oneflow kernel引入mlir执行。

## 编译期

### 1. FromGraphToMLIR
 - GraphToJob
 - JobToOneFlowDialect

### 2. OneFlowDialectToOKLDialect
通过三个Pass将OneFlow转换成okl的ir形式。
- extract-kernel-launch-tensor
- trim-return-to-void
- lower-to-okl
``` mlir
 module {
  func.func @wrap0(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
}
```
-extract-kernel-launch-tensor

将tensor的输入流转换为ctx中获取
``` mlir
module {
  func.func @wrap0(%arg0: !okl.launcher_ctx) -> (tensor<2xf32>, tensor<2xf32>) {
    %0 = "okl.arg_to_tensor"(%arg0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
    %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %2 = "oneflow.tanh"(%1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %3 = "okl.tensor_to_ret"(%arg0, %1) {index = 0 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
    %4 = "okl.tensor_to_ret"(%arg0, %2) {index = 1 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
    return %3, %4 : tensor<2xf32>, tensor<2xf32>
  }
}

```
-trim-return-to-void

将tensor的输出流删除掉
```mlir
module {
  func.func @wrap0(%arg0: !okl.launcher_ctx) {
    %0 = "okl.arg_to_tensor"(%arg0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
    %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %2 = "oneflow.tanh"(%1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %3 = "okl.tensor_to_ret"(%arg0, %1) {index = 0 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
    %4 = "okl.tensor_to_ret"(%arg0, %2) {index = 1 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
    return
  }
}
```
-lower-to-okl

将oneflow kernel op用okl wrapper_kernel封装起来，并通过okl op编译推导对应tensor流的信息。
```mlir
module {
  func.func @okl_func(%arg0: !okl.launcher_ctx) {
    "okl.wrapper_kernel"() ({
      %0 = "okl.arg_to_tensor"(%arg0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %1 = "oneflow.relu"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %2 = "okl.tensor_to_ret"(%arg0, %1) {index = 0 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 0 : i32} : () -> ()
    "okl.wrapper_kernel"() ({
      %0 = "okl.ret_to_tensor"(%arg0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %2 = "okl.tensor_to_ret"(%arg0, %1) {index = 1 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 1 : i32} : () -> ()
    return
  }
}
```
### 3. OKLDialectToLLVMDialect
通过四个Pass将OKL的IR转换为LLVMM的IR形式作为运行时的输入
- lower-launcher-to-llvm-ptr
- lower-okl-to-llvm-call
- reconcile-unrealized-casts
- convert-func-to-llvm

-lower-launcher-to-llvm-ptr

将ctx转换成一个llvm.ptr，通过llvm.ptr表示ctx的传递。
```mlir
module {
  func.func @okl_func(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
    "okl.wrapper_kernel"() ({
      %1 = "okl.arg_to_tensor"(%0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %2 = "oneflow.relu"(%1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %3 = "okl.tensor_to_ret"(%0, %2) {index = 0 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 0 : i32} : () -> ()
    "okl.wrapper_kernel"() ({
      %1 = "okl.ret_to_tensor"(%0) {index = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %2 = "oneflow.tanh"(%1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %3 = "okl.tensor_to_ret"(%0, %2) {index = 1 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {index = 1 : i32} : () -> ()
    return
  }
}
```
-lower-okl-to-llvm-call

将okl的wrapper_kernel转换成llvm的call调用。
```mlir
module {
  llvm.func @okl_llvm_func(!llvm.ptr<i8>, i64) attributes {llvm.emit_c_interface}
  func.func @okl_func(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr<i8> to !okl.launcher_ctx
    %1 = llvm.mlir.constant(0 : index) : i64
    llvm.call @okl_llvm_func(%arg0, %1) : (!llvm.ptr<i8>, i64) -> ()
    %2 = llvm.mlir.constant(1 : index) : i64
    llvm.call @okl_llvm_func(%arg0, %2) : (!llvm.ptr<i8>, i64) -> ()
    return
  }
}

```
-reconcile-unrealized-casts
-convert-func-to-llvm 

转换成可以直接运行的llvm IR
```mlir
module attributes {llvm.data_layout = ""} {
  llvm.func @okl_llvm_func(!llvm.ptr<i8>, i64) attributes {llvm.emit_c_interface}
  llvm.func @okl_func(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : index) : i64
    llvm.call @okl_llvm_func(%arg0, %0) : (!llvm.ptr<i8>, i64) -> ()
    %1 = llvm.mlir.constant(1 : index) : i64
    llvm.call @okl_llvm_func(%arg0, %1) : (!llvm.ptr<i8>, i64) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_okl_func(%arg0: !llvm.ptr<i8>) attributes {llvm.emit_c_interface} {
    llvm.call @okl_func(%arg0) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
}
```


## 运行时

OKLDialect IR不仅作为编译期最后一阶段的输入，同时作为运行时初始化时期资源的输入来初始化运行时的各种ctx，从而为计算期的计算做准备。

一个OKL的kernel包含了一整个子图。因此OKL的kernel需要管理若干有序子op的ctx资源。这些通过LauncherState来初始化创建，LauncherState中含有LauncherContext用来统一管理所有子Op的资源。

LauncherContext含有若干有序的CompileTimeWrapperContext一一对应其子Op未Infer前的ctx，以及若干RunTimeWrapperContext一一对应其子Op在Infer后的ctx。

下面为这两种Ctx所持有的资源。
```
class CompileTimeWrapperContext {
  std::shared_ptr<const RegContext> reg_ctx_;
};

class RunTimeWrapperContext : public CompileTimeWrapperContext {
  std::shared_ptr<ComputeContext> compute_ctx_;
  std::shared_ptr<InitContext> init_ctx_;

  std::shared_ptr<user_op::OpKernelState> kernel_state_;
  std::shared_ptr<user_op::OpKernelCache> kernel_cache_;
};
```
CompileTimeWrapperContext 主要是reg_ctx，以作为infer推导的必须输入。

RunTimeWrapperContext 包含所有子op运行时计算需要用的的资源，主要有compute_ctx以及state和cache。

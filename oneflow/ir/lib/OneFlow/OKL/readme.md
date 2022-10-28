OKL Dialect is the short name of OneFlow Kernel Launch Dialect

This Dialect is aimed at supporting a mediate abstract layer about kernel launch.

firstly, if `ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH` is set to 1, the end of roundtrip will wrap successive compute oneflow ops into a kernel_launch op.
```python
os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
```

such as:
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
to:

```mlir
module {
  func.func @wrap0(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) attributes {llvm.emit_c_interface} {
    %0 = "oneflow.relu"(%arg0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    %1 = "oneflow.tanh"(%0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    return %0, %1 : tensor<2xf32>, tensor<2xf32>
  }
  oneflow.job @GraphToRun_1(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_input.0.0_2", output_lbns = ["_GraphToRun_1_input.0.0_2/out"], scope_symbol_id = 30 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    %0:2 = oneflow.kernel_launch @wrap0(%output) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], mlir_assembly = "...", op_name = "wrap0", scope_symbol_id = 30 : i64} : (tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    %output_0 = "oneflow.output"(%0#1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_output.0.0_2", output_lbns = ["_GraphToRun_1_output.0.0_2/out"], scope_symbol_id = 30 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    oneflow.return %output_0 : tensor<2xf32>
  }
}
```

the function type of wrap is contained all tensor resources needed by the wrapped ops.

the resources will remap by okl_init_context which initialize okl.launcher_ctx for kernel compute.

okl dialect will be lowered to llvm dialect with a bunch of llvm.call whose callee is in liboneflow.so.

compute function will create a llvm engine run the llvm dialect.

// RUN: oneflow-opt %s \
// RUN: -fetch-from-launcher \
// RUN: | FileCheck %s

// CHECK-NOT: call @get_resources_type

module {
  func.func @okl_recycle(%arg0: !okl.launcher_ctx) {
    %0:2 = call @get_resources_type_0(%arg0) : (!okl.launcher_ctx) -> (!okl.reg_ctx, !okl.reg_ctx)
    %1:2 = call @get_resources_type_1(%arg0) : (!okl.launcher_ctx) -> (!okl.run_ctx, !okl.run_ctx)
    %2:2 = call @get_resources_type_2(%arg0) : (!okl.launcher_ctx) -> (!okl.kernel, !okl.kernel)
    "okl.destroy_reg_ctx"(%0#0) : (!okl.reg_ctx) -> ()
    "okl.destroy_reg_ctx"(%0#1) : (!okl.reg_ctx) -> ()
    "okl.destroy_run_ctx"(%1#0) : (!okl.run_ctx) -> ()
    "okl.destroy_run_ctx"(%1#1) : (!okl.run_ctx) -> ()
    return
  }
  func.func @okl_compute(%arg0: !okl.launcher_ctx) {
    %0:2 = call @get_resources_type_0(%arg0) : (!okl.launcher_ctx) -> (!okl.reg_ctx, !okl.reg_ctx)
    %1:2 = call @get_resources_type_1(%arg0) : (!okl.launcher_ctx) -> (!okl.run_ctx, !okl.run_ctx)
    %2:2 = call @get_resources_type_2(%arg0) : (!okl.launcher_ctx) -> (!okl.kernel, !okl.kernel)
    "okl.launch"(%1#0, %2#0) : (!okl.run_ctx, !okl.kernel) -> ()
    "okl.launch"(%1#1, %2#1) : (!okl.run_ctx, !okl.kernel) -> ()
    return
  }
  func.func @okl_init_context(%arg0: !okl.launcher_ctx) {
    %0 = "okl.build_reg_ctx"() ({
    ^bb0(%arg1: !okl.launcher_ctx):
      %6 = "okl.get_tensor_from_arg"(%arg1) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %7 = "oneflow.relu"(%6) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "relu-0", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %8 = "okl.get_tensor_as_ret"(%arg1, %7) {index = 0 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {function_type = (!okl.launcher_ctx) -> ()} : () -> !okl.reg_ctx
    %1 = "okl.build_reg_ctx"() ({
    ^bb0(%arg1: !okl.launcher_ctx):
      %6 = "okl.get_tensor_from_ret"(%arg1) {index = 0 : i32, tensor_type = 0 : i32} : (!okl.launcher_ctx) -> tensor<2xf32>
      %7 = "oneflow.tanh"(%6) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "tanh-1", scope_symbol_id = 12 : i64} : (tensor<2xf32>) -> tensor<2xf32>
      %8 = "okl.get_tensor_as_ret"(%arg1, %7) {index = 1 : i32, tensor_type = 2 : i32} : (!okl.launcher_ctx, tensor<2xf32>) -> tensor<2xf32>
      okl.return
    }) {function_type = (!okl.launcher_ctx) -> ()} : () -> !okl.reg_ctx
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

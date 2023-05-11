// RUN: oneflow-opt %s \
// RUN: -ofjob-to-func \
// RUN: -convert-to-signless-for-tosa \
// RUN: -lower-oneflow-to-tosa="full=0 lower-job=0" \
// RUN: --tosa-make-broadcastable \
// RUN: -lower-oneflow-to-linalg \
// RUN: -tosa-to-tensor \
// RUN: | oneflow-opt -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg))" \
// RUN: | oneflow-opt -linalg-fuse-elementwise-ops \
// RUN: -func-to-ofjob \
// RUN: | oneflow-opt -pass-pipeline="builtin.module(oneflow.job(outline-jit-function{compile-to-llvm=0}))" \
// RUN: | oneflow-opt -canonicalize \
// RUN: | FileCheck --dump-input=always %s

// CHECK: linalg.generic
// CHECK: oneflow.mlir_jit
// CHECK-NOT: oneflow.softmax

oneflow.job @GraphToRun_11(%arg0: tensor<2x256x1280xf16>, %arg1: tensor<2x77x1280xf16>, %arg2: tensor<2x77x1280xf16>) -> tensor<2x256x1280xf16> {
  %output = "oneflow.input"(%arg0) {data_type = 9 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_11_input.0.0_2", output_lbns = ["_GraphToRun_11_input.0.0_2/out"], scope_symbol_id = 681 : i64, shape = [2 : si64, 256 : si64, 1280 : si64]} : (tensor<2x256x1280xf16>) -> tensor<2x256x1280xf16>
  %output_0 = "oneflow.input"(%arg1) {data_type = 9 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_11_input.0.1_3", output_lbns = ["_GraphToRun_11_input.0.1_3/out"], scope_symbol_id = 681 : i64, shape = [2 : si64, 77 : si64, 1280 : si64]} : (tensor<2x77x1280xf16>) -> tensor<2x77x1280xf16>
  %output_1 = "oneflow.input"(%arg2) {data_type = 9 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_11_input.0.2_4", output_lbns = ["_GraphToRun_11_input.0.2_4/out"], scope_symbol_id = 681 : i64, shape = [2 : si64, 77 : si64, 1280 : si64]} : (tensor<2x77x1280xf16>) -> tensor<2x77x1280xf16>
  %0 = "oneflow.reshape"(%output) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-0", scope_symbol_id = 681 : i64, shape = [2 : si64, 256 : si64, 8 : si64, 160 : si64]} : (tensor<2x256x1280xf16>) -> tensor<2x256x8x160xf16>
  %1 = "oneflow.reshape"(%output_0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-2", scope_symbol_id = 681 : i64, shape = [2 : si64, 77 : si64, 8 : si64, 160 : si64]} : (tensor<2x77x1280xf16>) -> tensor<2x77x8x160xf16>
  %2 = "oneflow.reshape"(%output_1) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-4", scope_symbol_id = 681 : i64, shape = [2 : si64, 77 : si64, 8 : si64, 160 : si64]} : (tensor<2x77x1280xf16>) -> tensor<2x77x8x160xf16>
  %3 = "oneflow.transpose"(%0) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "transpose-1", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 681 : i64} : (tensor<2x256x8x160xf16>) -> tensor<2x8x256x160xf16>
  %4 = "oneflow.transpose"(%1) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "transpose-3", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 681 : i64} : (tensor<2x77x8x160xf16>) -> tensor<2x8x77x160xf16>
  %5 = "oneflow.transpose"(%2) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "transpose-5", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 681 : i64} : (tensor<2x77x8x160xf16>) -> tensor<2x8x77x160xf16>
  %6 = "oneflow.reshape"(%3) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-6", scope_symbol_id = 681 : i64, shape = [16 : si64, 256 : si64, 160 : si64]} : (tensor<2x8x256x160xf16>) -> tensor<16x256x160xf16>
  %7 = "oneflow.reshape"(%4) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-7", scope_symbol_id = 681 : i64, shape = [16 : si64, 77 : si64, 160 : si64]} : (tensor<2x8x77x160xf16>) -> tensor<16x77x160xf16>
  %8 = "oneflow.reshape"(%5) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-9", scope_symbol_id = 681 : i64, shape = [16 : si64, 77 : si64, 160 : si64]} : (tensor<2x8x77x160xf16>) -> tensor<16x77x160xf16>
  %9 = "oneflow.transpose"(%7) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "transpose-8", perm = [0 : si32, 2 : si32, 1 : si32], scope_symbol_id = 681 : i64} : (tensor<16x77x160xf16>) -> tensor<16x160x77xf16>
  %10 = "oneflow.batch_matmul"(%6, %9) {alpha = 0.079056941504209485 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "batch_matmul-11", scope_symbol_id = 681 : i64, transpose_a = false, transpose_b = false} : (tensor<16x256x160xf16>, tensor<16x160x77xf16>) -> tensor<16x256x77xf16>
  %11 = "oneflow.softmax"(%10) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "softmax-12", scope_symbol_id = 681 : i64} : (tensor<16x256x77xf16>) -> tensor<16x256x77xf16>
  %12 = "oneflow.batch_matmul"(%11, %8) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "batch_matmul-13", scope_symbol_id = 681 : i64, transpose_a = false, transpose_b = false} : (tensor<16x256x77xf16>, tensor<16x77x160xf16>) -> tensor<16x256x160xf16>
  %13 = "oneflow.reshape"(%12) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-14", scope_symbol_id = 681 : i64, shape = [2 : si64, 8 : si64, 256 : si64, 160 : si64]} : (tensor<16x256x160xf16>) -> tensor<2x8x256x160xf16>
  %14 = "oneflow.transpose"(%13) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "transpose-15", perm = [0 : si32, 2 : si32, 1 : si32, 3 : si32], scope_symbol_id = 681 : i64} : (tensor<2x8x256x160xf16>) -> tensor<2x256x8x160xf16>
  %15 = "oneflow.reshape"(%14) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "reshape-16", scope_symbol_id = 681 : i64, shape = [2 : si64, 256 : si64, 1280 : si64]} : (tensor<2x256x8x160xf16>) -> tensor<2x256x1280xf16>
  %output_2 = "oneflow.output"(%15) {data_type = 9 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_11_output.0.0_2", output_lbns = ["_GraphToRun_11_output.0.0_2/out"], scope_symbol_id = 681 : i64, shape = [2 : si64, 256 : si64, 1280 : si64]} : (tensor<2x256x1280xf16>) -> tensor<2x256x1280xf16>
  oneflow.return %output_2 : tensor<2x256x1280xf16>
}

// CHECK: oneflow.mlir_jit
// CHECK-NOT: oneflow.cast
oneflow.job @GraphToRun_1(%arg0: tensor<2x5xsi64>, %arg1: tensor<1xf32>) -> tensor<2x5xf32> {
  %output = "oneflow.input"(%arg0) {data_type = 6 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_input.0.0_2", output_lbns = ["_GraphToRun_1_input.0.0_2/out"], scope_symbol_id = 34 : i64, shape = [2 : si64, 5 : si64]} : (tensor<2x5xsi64>) -> tensor<2x5xsi64>
  %output_0 = "oneflow.input"(%arg1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_input.0.1_3", output_lbns = ["_GraphToRun_1_input.0.1_3/out"], scope_symbol_id = 34 : i64, shape = [1 : si64]} : (tensor<1xf32>) -> tensor<1xf32>
  %0 = "oneflow.cast"(%output) {device_name = ["@0:0"], device_tag = "cpu", dtype = 2 : i32, hierarchy = [1], op_name = "fw-cast-0", pin_memory = false, scope_symbol_id = 41 : i64} : (tensor<2x5xsi64>) -> tensor<2x5xf32>
  %1 = "oneflow.scalar_mul_by_tensor"(%0, %output_0) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "fw-broadcast_mul-1-mlir-gen-2", scope_symbol_id = 41 : i64} : (tensor<2x5xf32>, tensor<1xf32>) -> tensor<2x5xf32>
  %output_1 = "oneflow.output"(%1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_1_output.0.0_2", output_lbns = ["_GraphToRun_1_output.0.0_2/out"], scope_symbol_id = 34 : i64, shape = [2 : si64, 5 : si64]} : (tensor<2x5xf32>) -> tensor<2x5xf32>
  oneflow.return %output_1 : tensor<2x5xf32>
}

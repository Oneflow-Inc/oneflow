// RUN: oneflow-opt %s -oneflow-request-device-wrappers | \
// RUN: FileCheck %s

// RUN: oneflow-opt %s -oneflow-request-device-wrappers="device-type=cpu" | \
// RUN: FileCheck %s --check-prefix=CHECK-CPU

// CHECK: oneflow.gpu
// CHECK-CPU: oneflow.cpu
module  {
  func.func @test(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xsi64> {
    %1, %indice = "oneflow.max_pool_2d"(%arg0) {
      ceil_mode = false,
      data_format = "channels_first",
      device_name = ["@0:0"],
      device_tag = "cpu",
      dilation = [1 : si32, 1 : si32],
      hierarchy = [1],
      kernel_size = [3 : si32, 3 : si32],
      op_name = "model.maxpool-max_pool_2d-3",
      padding = [1 : si32, 1 : si32],
      return_indices = false,
      scope_symbol_id = 49 : i64,
      stride = [2 : si32, 2 : si32]
    } : (tensor<1x64x112x112xf32>) -> (tensor<1x64x56x56xf32>, tensor<1x64x56x56xsi64>)
    return %indice : tensor<1x64x56x56xsi64>
  }
}
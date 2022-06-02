// RUN: oneflow-opt -lower-oneflow-to-tosa %s | FileCheck %s

module {
  oneflow.job @GraphModule_0(%arg: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32> {
    %y, %indice = "oneflow.max_pool_2d"(%arg) {ceil_mode = false, data_format = "channels_first", device_name = ["@0:0"], device_tag = "cpu",
     dilation = [1 : si32, 1 : si32], hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "fw.model.maxpool-max_pool_2d-4", output_lbns = ["fw.model.maxpool-max_pool_2d-4/y_0", "fw.model.maxpool-max_pool_2d-4/indice_0"],
     padding = [1 : si32, 1 : si32], return_indices = false, scope_symbol_id = 4611686018427502591 : i64, stride = [2 : si32, 2 : si32]} :
     (tensor<1x64x112x112xf32>) -> (tensor<1x64x56x56xf32>, tensor<1x64x56x56xi64>)
// CHECK:      return %0 : tensor<1x64x56x56xf32>
    oneflow.return %y :  tensor<1x64x56x56xf32>
  }
}

module {
  oneflow.job @GraphModule_0(%arg: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xi64> {
    %y, %indice = "oneflow.max_pool_2d"(%arg) {ceil_mode = false, data_format = "channels_first", device_name = ["@0:0"], device_tag = "cpu",
     dilation = [1 : si32, 1 : si32], hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "fw.model.maxpool-max_pool_2d-4", output_lbns = ["fw.model.maxpool-max_pool_2d-4/y_0", "fw.model.maxpool-max_pool_2d-4/indice_0"],
     padding = [1 : si32, 1 : si32], return_indices = false, scope_symbol_id = 4611686018427502591 : i64, stride = [2 : si32, 2 : si32]} :
     (tensor<1x64x112x112xf32>) -> (tensor<1x64x56x56xf32>, tensor<1x64x56x56xi64>)
// CHECK:      return %1 : tensor<1x64x56x56xi64>
    oneflow.return %indice :  tensor<1x64x56x56xi64>
  }
}

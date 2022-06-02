// RUN: oneflow-opt -lower-oneflow-to-tosa %s | FileCheck %s

module {
  oneflow.job @GraphModule_0(%arg0: tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32> {
// CHECK:    %0 = "tosa.avg_pool2d"(%arg0) {kernel = [7, 7], pad = [0, 0, 0, 0], stride = [7, 7]} : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32>
    %out = "oneflow.avg_pool_2d"(%arg0) {ceil_mode = false, count_include_pad = true, data_format = "channels_first", device_name = ["@0:0"],
     device_tag = "cpu", divisor_override = 0 : si32, hierarchy = [1], kernel_size = [7 : si32, 7 : si32],
      op_name = "model.avgpool-avg_pool_2d-172", output_lbns = ["model.avgpool-avg_pool_2d-172/y_0"], padding = [0 : si32, 0 : si32],
       scope_symbol_id = 4611686018430775295 : i64, stride = [7 : si32, 7 : si32]}
       : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32>
    oneflow.return %out : tensor<1x2048x1x1xf32>
  }
}

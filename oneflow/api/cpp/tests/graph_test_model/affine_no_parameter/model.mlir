module  {
  oneflow.job @MyGraph_1(%arg0: tensor<1x3xf32>, %arg1: tensor<3x2xf32>, %arg2: tensor<2xf32>) -> tensor<1x2xf32> {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_MyGraph_1-input_0", output_lbns = ["_MyGraph_1-input_0/out"], scope_symbol_id = 4611686018427527167 : i64, shape = [1 : si64, 3 : si64]} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    %output_0 = "oneflow.input"(%arg1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_MyGraph_1-input_1", output_lbns = ["_MyGraph_1-input_1/out"], scope_symbol_id = 4611686018427527167 : i64, shape = [3 : si64, 2 : si64]} : (tensor<3x2xf32>) -> tensor<3x2xf32>
    %output_1 = "oneflow.input"(%arg2) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_MyGraph_1-input_2", output_lbns = ["_MyGraph_1-input_2/out"], scope_symbol_id = 4611686018427527167 : i64, shape = [2 : si64]} : (tensor<2xf32>) -> tensor<2xf32>
    %0 = "oneflow.matmul"(%output, %output_0) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "model-matmul_0", output_lbns = ["model-matmul_0/out_0"], scope_symbol_id = 4611686018427535359 : i64, transpose_a = false, transpose_b = false} : (tensor<1x3xf32>, tensor<3x2xf32>) -> tensor<1x2xf32>
    %1 = "oneflow.broadcast_add"(%0, %output_1) {device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], op_name = "model-broadcast_add_1", output_lbns = ["model-broadcast_add_1/z_0"], scope_symbol_id = 4611686018427535359 : i64} : (tensor<1x2xf32>, tensor<2xf32>) -> tensor<1x2xf32>
    %output_2 = "oneflow.output"(%1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_MyGraph_1-output_0", output_lbns = ["_MyGraph_1-output_0/out"], scope_symbol_id = 4611686018427527167 : i64, shape = [1 : si64, 2 : si64]} : (tensor<1x2xf32>) -> tensor<1x2xf32>
    oneflow.return %output_2 : tensor<1x2xf32>
  }
}

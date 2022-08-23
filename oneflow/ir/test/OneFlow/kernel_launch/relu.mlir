// RUN: oneflow-opt %s \
// RUN: -kernel-launch-function
module {

  oneflow.job @GraphToRun_0(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    %output, %ctrl_output = "oneflow.input"(%arg0) {
        data_type = 2 : i32,
        device_name = ["@0:0"],
        device_tag = "cpu", hierarchy = [1],
        is_dynamic = false,
        nd_sbp = ["B"],
        op_name = "_GraphToRun_0_input.0.0_2",
        output_lbns = ["_GraphToRun_0_input.0.0_2/out"],
        scope_symbol_id = 12 : i64, shape = [1 : si64]
    } : (tensor<1xf32>) -> (tensor<1xf32>, tensor<i1>)

    %data_output, %ctrl_output_0 = "oneflow.user"(%output) {
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        input_sizes = [1 : i32],
        op_name = "relu-0",
        op_type_name = "relu",
        operand_segment_sizes = dense<[1, 0]> : vector<2xi32>,
        output_lbns = ["relu-0/y_0"],
        output_sizes = [1 : i32],
        result_segment_sizes = dense<1> : vector<2xi32>,
        scope_symbol_id = 12 : i64
    } : (tensor<1xf32>) -> (tensor<1xf32>,  tensor<i1>)
    %data_output_1, %ctrl_output_2 = "oneflow.user"(%data_output) {
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        input_sizes = [1 : i32],
        op_name = "relu-1",
        op_type_name = "relu",
        operand_segment_sizes = dense<[1, 0]> : vector<2xi32>,
        output_lbns = ["relu-1/y_0"],
        output_sizes = [1 : i32],
        result_segment_sizes = dense<1> : vector<2xi32>,
        scope_symbol_id = 12 : i64
    } : (tensor<1xf32>) -> (tensor<1xf32>, tensor<i1>)
    %output_3, %ctrl_output_4 = "oneflow.output"(%data_output_1) {
        data_type = 2 : i32,
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        is_dynamic = false,
        nd_sbp = ["B"],
        op_name = "_GraphToRun_0_output.0.0_2",
        output_lbns = ["_GraphToRun_0_output.0.0_2/out"],
        scope_symbol_id = 12 : i64,
        shape = [1 : si64]
    } : (tensor<1xf32>) -> (tensor<1xf32>, tensor<i1>)
    oneflow.return %output_3 : tensor<1xf32>
  }

}

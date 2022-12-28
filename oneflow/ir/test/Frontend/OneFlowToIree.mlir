// RUN: oneflow-opt %s \
// RUN: -split-input-file \
// RUN: -lower-oneflow-to-tosa \
// RUN: -tosa-make-broadcastable \
// RUN: -verify-diagnostics -o - \
// RUN: | python3 -m iree.compiler.tools.scripts.ireec \
// RUN: --iree-input-type=tosa \
// RUN: --iree-vm-bytecode-module-output-format=flatbuffer-binary \
// RUN: -iree-hal-target-backends=dylib-llvm-aot \
// RUN: -iree-mlir-to-vm-bytecode-module -


oneflow.job @test_func(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
    oneflow.return %arg0 : tensor<1xf32>
}


oneflow.job @test_input(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
    %res = "oneflow.input"(%arg0)
    {
        data_type = 2 : i32,
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        is_dynamic = false,
        nd_sbp = ["B"],
        op_name = "",
        output_lbns = [""],
        scope_symbol_id = 4611686018427412479 : i64,
        shape = [1 : si64]
    } : (tensor<1xf32>) -> tensor<1xf32>
    oneflow.return %res : tensor<1xf32>
}


oneflow.job @test_output(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
    %res = "oneflow.output"(%arg0)
    {
        data_type = 2 : i32,
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        is_dynamic = false,
        nd_sbp = ["B"],
        op_name = "",
        output_lbns = [""],
        scope_symbol_id = 4611686018427412479 : i64,
        shape = [1 : si64]
    } : (tensor<1xf32>) -> tensor<1xf32>
    oneflow.return %res : tensor<1xf32>
}


oneflow.job @test_variable() -> tensor<64x3x7x7xf32>
{
    %res = "oneflow.variable"() {
        data_type = 2 : i32,
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        parallel = #sbp.parallel<[] -> [#sbp.B]>,
        op_name = "fw.model.conv1.weight",
        output_lbns = ["fw.model.conv1.weight/out"],
        scope_symbol_id = 4611686018427432959 : i64,
        shape = [64 : si64, 3 : si64, 7 : si64, 7 : si64]
    } : () -> tensor<64x3x7x7xf32>
    oneflow.return %res : tensor<64x3x7x7xf32>
}


oneflow.job @test_add_n2(%arg0: tensor<1x7x7xf32>, %arg1: tensor<1x7x7xf32>) -> tensor<1x7x7xf32>
{
    %res = "oneflow.add_n2"(%arg0, %arg1)
    {
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        op_name = "",
        op_type_name = "add_n",
        output_lbns = [""],
        scope_symbol_id = 4611686018431205375 : i64
    } : (tensor<1x7x7xf32>, tensor<1x7x7xf32>) -> tensor<1x7x7xf32>
    oneflow.return %res: tensor<1x7x7xf32>
}


oneflow.job @test_broadcast_add(%arg0: tensor<1x1000xf32>, %arg1: tensor<1000xf32>) -> tensor<1x1000xf32>
{
    %res = "oneflow.broadcast_add"(%arg0, %arg1)
    {
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        op_name = "",
        output_lbns = [""],
        scope_symbol_id = 4611686018431234047 : i64
    } : (tensor<1x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
    oneflow.return %res : tensor<1x1000xf32>
}


oneflow.job @test_max_pool_2d(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
{
    %y, %indice = "oneflow.max_pool_2d"(%arg0)
    {
        ceil_mode = false,
        data_format = "channels_first",
        device_name = ["@0:0"],
        device_tag = "cpu",
        dilation = [1 : si32, 1 : si32],
        hierarchy = [1], kernel_size = [3 : si32, 3 : si32],
        op_name = "",
        output_lbns = ["", ""],
        padding = [1 : si32, 1 : si32],
        return_indices = false,
        scope_symbol_id = 4611686018427502591 : i64,
        stride = [2 : si32, 2 : si32]
    } : (tensor<1x64x112x112xf32>) -> (tensor<1x64x56x56xf32>, tensor<1x64x56x56xi64>)
    oneflow.return %y :  tensor<1x64x56x56xf32>
}


oneflow.job @test_avg_pool_2d(%arg0: tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32>
{
    %res = "oneflow.avg_pool_2d"(%arg0)
    {
        ceil_mode = false,
        count_include_pad = true,
        data_format = "channels_first",
        device_name = ["@0:0"],
        device_tag = "cpu",
        divisor_override = 0 : si32,
        hierarchy = [1],
        kernel_size = [7 : si32, 7 : si32],
        op_name = "model.avgpool-avg_pool_2d-172",
        output_lbns = ["model.avgpool-avg_pool_2d-172/y_0"],
        padding = [0 : si32, 0 : si32],
        scope_symbol_id = 4611686018430775295 : i64,
        stride = [7 : si32, 7 : si32]
    } : (tensor<1x2048x7x7xf32>) -> tensor<1x2048x1x1xf32>
    oneflow.return %res: tensor<1x2048x1x1xf32>
}


oneflow.job @test_conv2d(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<5x3x1x1xf32>) -> tensor<1x5x224x224xf32>
{
    %res = "oneflow.conv2d"(%arg0, %arg1)
    {
        data_format = "channels_first",
        device_name = ["@0:0"],
        device_tag = "cpu",
        dilation_rate = [1 : si32, 1 : si32],
        filters = 512 : si32,
        groups = 1 : si32,
        hierarchy = [1],
        kernel_size = [1 : si32, 1 : si32],
        op_name = "",
        operand_segment_sizes = dense<[1, 1, 0, 0]> : vector<4xi32>,
        output_lbns = [""],
        padding_before = [0 : si32, 0 : si32],
        scope_symbol_id = 4611686018431012863 : i64,
        strides = [1 : si32, 1 : si32]
    } : (tensor<1x3x224x224xf32>, tensor<5x3x1x1xf32>) -> tensor<1x5x224x224xf32>
    oneflow.return %res : tensor<1x5x224x224xf32>
}


oneflow.job @test_matmul(%arg0: tensor<1x2048xf32>, %arg1: tensor<1000x2048xf32>) ->tensor<1x1000xf32>
{
    %res = "oneflow.matmul"(%arg0, %arg1)
    {
        alpha = 1.000000e+00 : f64,
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        op_name = "",
        output_lbns = [""],
        scope_symbol_id = 4611686018431234047 : i64,
        transpose_a = false,
        transpose_b = true
    } : (tensor<1x2048xf32>, tensor<1000x2048xf32>) -> tensor<1x1000xf32>
    oneflow.return %res : tensor<1x1000xf32>
}


oneflow.job @test_relu(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    %res = "oneflow.relu"(%arg0)
    {
        device_name = ["@0:0"],
        device_tag = "cpu",
        hierarchy = [1],
        op_name = "",
        output_lbns = [""],
        scope_symbol_id = 4611686018427424767 : i64
    } : (tensor<1xf32>) -> tensor<1xf32>
    oneflow.return %res : tensor<1xf32>
}

oneflow.job @test_bn(
%x:               tensor<1x64x112x112xf32>,
%moving_mean:     tensor<64xf32>,
%moving_variance: tensor<64xf32>,
%gamma:           tensor<64xf32>,
%beta:            tensor<64xf32>) -> tensor<1x64x112x112xf32>
{
    %y, %mean, %inv_variance = "oneflow.normalization"(%x, %moving_mean, %moving_variance, %gamma, %beta)
    {
        axis = 1 : si32,
        device_name = ["@0:0"],
        device_tag = "cpu",
        epsilon = 9.99999974E-6 : f32,
        hierarchy = [1],
        momentum = 0.899999976 : f32,
        op_name = "",
        operand_segment_sizes = dense<[1, 1, 1, 1, 1, 0]> : vector<6xi32>,
        output_lbns = ["", "", ""],
        result_segment_sizes = dense<1> : vector<3xi32>,
        scope_symbol_id = 4611686018427453439 : i64,
        training = true
    } : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>)
    oneflow.return %y: tensor<1x64x112x112xf32>
}

oneflow.job @test_bn_infer(
%x:               tensor<1x64x112x112xf32>,
%moving_mean:     tensor<64xf32>,
%moving_variance: tensor<64xf32>,
%gamma:           tensor<64xf32>,
%beta:            tensor<64xf32>) -> tensor<1x64x112x112xf32>
{
    %y = "oneflow.normalization_infer"(%x, %moving_mean, %moving_variance, %gamma, %beta)
    {
        axis = 1 : si32,
        device_name = ["@0:0"],
        device_tag = "cpu",
        epsilon = 9.99999974E-6 : f32,
        hierarchy = [1],
        momentum = 0.899999976 : f32,
        op_name = "",
        operand_segment_sizes = dense<[1, 1, 1, 1, 1, 0]> : vector<6xi32>,
        output_lbns = ["", "", ""],
        result_segment_sizes = dense<1> : vector<3xi32>,
        scope_symbol_id = 4611686018427453439 : i64,
        training = true
    } : (tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    oneflow.return %y: tensor<1x64x112x112xf32>
}

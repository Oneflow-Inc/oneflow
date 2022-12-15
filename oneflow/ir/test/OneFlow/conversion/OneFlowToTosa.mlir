// RUN: oneflow-opt %s \
// RUN: -split-input-file \
// RUN: -lower-oneflow-to-tosa \
// RUN: -verify-diagnostics -o - \
// RUN: | FileCheck %s


// CHECK-LABEL: test_func
// CHECK: return [[V0:%.+]] : tensor<1xf32>
oneflow.job @test_func(%arg0: tensor<1xf32>) -> tensor<1xf32>
{
    oneflow.return %arg0 : tensor<1xf32>
}


// CHECK-LABEL: test_input
// CHECK: return [[V0:%.+]] : tensor<1xf32>
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


// CHECK-LABEL: test_output
// CHECK: return [[V0:%.+]] : tensor<1xf32>
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


// CHECK-LABEL: test_variable
// CHECK: [[V0:%.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<64x3x7x7xf32>} : () -> tensor<64x3x7x7xf32>
// CHECK: return [[V0]] : tensor<64x3x7x7xf32>
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


//CHECK-LABEL: test_add_n2
//CHECK: [[V0:%.+]] = "tosa.add"(%arg0, %arg1) : (tensor<1x7x7xf32>, tensor<1x7x7xf32>) -> tensor<1x7x7xf32>
//CHECK: return [[V0]] : tensor<1x7x7xf32>
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


//CHECK-LABEL: test_broadcast_add
//CHECK: [[V0:%.+]] = "tosa.add"(%arg0, %arg1) : (tensor<1x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
//CHECK: return [[V0]] : tensor<1x1000xf32>
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


//CHECK-LABEL: test_max_pool_2d
//CHECK: [[V0:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
//CHECK: [[V1:%.+]] = "tosa.transpose"(%arg0, [[V0]]) : (tensor<1x64x112x112xf32>, tensor<4xi32>) -> tensor<1x112x112x64xf32>
//CHECK: [[V2:%.+]] = "tosa.max_pool2d"([[V1]]) {kernel = [3, 3], pad = [1, 1, 1, 1], stride = [2, 2]} : (tensor<1x112x112x64xf32>) -> tensor<1x56x56x64xf32>
//CHECK: [[V3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
//CHECK: [[V4:%.+]] = "tosa.transpose"([[V2]], [[V3]]) : (tensor<1x56x56x64xf32>, tensor<4xi32>) -> tensor<1x64x56x56xf32>
//CHECK: [[V5:%.+]] = "tosa.const"() {value = dense<0> : tensor<1x64x56x56xi64>} : () -> tensor<1x64x56x56xi64>
//CHECK: return [[V4]] : tensor<1x64x56x56xf32>
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


//CHECK-LABEL: test_avg_pool_2d
//CHECK: [[V0:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
//CHECK: [[V1:%.+]] = "tosa.transpose"(%arg0, [[V0]]) : (tensor<1x2048x7x7xf32>, tensor<4xi32>) -> tensor<1x7x7x2048xf32>
//CHECK: [[V2:%.+]] = "tosa.avg_pool2d"([[V1]]) {kernel = [7, 7], pad = [0, 0, 0, 0], stride = [7, 7]} : (tensor<1x7x7x2048xf32>) -> tensor<1x1x1x2048xf32>
//CHECK: [[V3:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
//CHECK: [[V4:%.+]] = "tosa.transpose"([[V2]], [[V3]]) : (tensor<1x1x1x2048xf32>, tensor<4xi32>) -> tensor<1x2048x1x1xf32>
//CHECK: return [[V4]] : tensor<1x2048x1x1xf32>
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


//CHECK-LABEL: test_conv2d
//CHECK: [[V0:%.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<5xf32>} : () -> tensor<5xf32>
//CHECK: [[V1:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
//CHECK: [[V2:%.+]] = "tosa.transpose"(%arg0, [[V1]]) : (tensor<1x3x224x224xf32>, tensor<4xi32>) -> tensor<1x224x224x3xf32>
//CHECK: [[V3:%.+]] = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
//CHECK: [[V4:%.+]] = "tosa.transpose"(%arg1, [[V3]]) : (tensor<5x3x1x1xf32>, tensor<4xi32>) -> tensor<5x1x1x3xf32>
//CHECK: [[V5:%.+]] = "tosa.conv2d"([[V2]], [[V4]], [[V0]]) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x224x224x3xf32>, tensor<5x1x1x3xf32>, tensor<5xf32>) -> tensor<1x224x224x5xf32>
//CHECK: [[V6:%.+]] = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
//CHECK: [[V7:%.+]] = "tosa.transpose"([[V5]], [[V6]]) : (tensor<1x224x224x5xf32>, tensor<4xi32>) -> tensor<1x5x224x224xf32>
//CHECK: return [[V7]] : tensor<1x5x224x224xf32>
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


//CHECK-LABEL: test_matmul
//CHECK: [[V0:%.+]] = "tosa.reshape"(%arg0) {new_shape = [1, 1, 2048]} : (tensor<1x2048xf32>) -> tensor<1x1x2048xf32>
//CHECK: [[V1:%.+]] = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
//CHECK: [[V2:%.+]] = "tosa.transpose"(%arg1, [[V1]]) : (tensor<1000x2048xf32>, tensor<2xi32>) -> tensor<2048x1000xf32>
//CHECK: [[V3:%.+]] = "tosa.reshape"([[V2]]) {new_shape = [1, 2048, 1000]} : (tensor<2048x1000xf32>) -> tensor<1x2048x1000xf32>
//CHECK: [[V4:%.+]] = "tosa.matmul"([[V0]], [[V3]]) : (tensor<1x1x2048xf32>, tensor<1x2048x1000xf32>) -> tensor<1x1x1000xf32>
//CHECK: [[V5:%.+]] = "tosa.reshape"([[V4]]) {new_shape = [1, 1000]} : (tensor<1x1x1000xf32>) -> tensor<1x1000xf32>
//CHECK: return [[V5]] : tensor<1x1000xf32>
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


//CHECK-LABEL: test_relu
//CHECK: [[V0:%.+]] = "tosa.reluN"(%arg0) {max_fp = 3.40282347E+38 : f32, max_int = 9223372036854775807 : i64} : (tensor<1xf32>) -> tensor<1xf32>
//CHECK: return [[V0]] : tensor<1xf32>
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

//CHECK-LABEL: test_bn
//CHECK: [[V0:%.+]] = "tosa.const"() {value = dense<9.99999974E-6> : tensor<f32>} : () -> tensor<f32>
//CHECK: [[V1:%.+]] = "tosa.reshape"(%arg1) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK: [[V2:%.+]] = "tosa.reshape"(%arg2) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK: [[V3:%.+]] = "tosa.reshape"(%arg3) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK: [[V4:%.+]] = "tosa.reshape"(%arg4) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK: [[V5:%.+]] = "tosa.sub"(%arg0, [[V1]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK: [[V6:%.+]] = "tosa.add"([[V2]], [[V0]]) : (tensor<64x1x1xf32>, tensor<f32>) -> tensor<64x1x1xf32>
//CHECK: [[V7:%.+]] = "tosa.rsqrt"([[V6]]) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
//CHECK: [[V8:%.+]] = "tosa.mul"([[V5]], [[V7]]) {shift = 0 : i32} : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK: [[V9:%.+]] = "tosa.mul"([[V8]], [[V3]]) {shift = 0 : i32} : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK: [[V10:%.+]] = "tosa.add"([[V9]], [[V4]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK: return [[V10]] : tensor<1x64x112x112xf32>
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

//CHECK-LABEL: test_bn_infer
//CHECK: [[V0:%.+]] = "tosa.const"() {value = dense<9.99999974E-6> : tensor<f32>} : () -> tensor<f32>
//CHECK: [[V1:%.+]] = "tosa.reshape"(%arg1) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK: [[V2:%.+]] = "tosa.reshape"(%arg2) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK: [[V3:%.+]] = "tosa.reshape"(%arg3) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK: [[V4:%.+]] = "tosa.reshape"(%arg4) {new_shape = [64, 1, 1]} : (tensor<64xf32>) -> tensor<64x1x1xf32>
//CHECK: [[V5:%.+]] = "tosa.sub"(%arg0, [[V1]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK: [[V6:%.+]] = "tosa.add"([[V2]], [[V0]]) : (tensor<64x1x1xf32>, tensor<f32>) -> tensor<64x1x1xf32>
//CHECK: [[V7:%.+]] = "tosa.rsqrt"([[V6]]) : (tensor<64x1x1xf32>) -> tensor<64x1x1xf32>
//CHECK: [[V8:%.+]] = "tosa.mul"([[V5]], [[V7]]) {shift = 0 : i32} : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK: [[V9:%.+]] = "tosa.mul"([[V8]], [[V3]]) {shift = 0 : i32} : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK: [[V10:%.+]] = "tosa.add"([[V9]], [[V4]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
//CHECK: return [[V10]] : tensor<1x64x112x112xf32>
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

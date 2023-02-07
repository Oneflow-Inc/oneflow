module {
  func.func @okm_wrap_subgraph0() {
    %0 = "okm.arg_to_memref"() {index = 0 : i32} : () -> memref<1x4x64x64xf16>
    %1 = "okm.arg_to_memref"() {index = 1 : i32} : () -> memref<4x1x1x4xf16>
    %2 = "okm.arg_to_memref"() {index = 2 : i32} : () -> memref<4xf16>
    %3 = "okm.arg_to_memref"() {index = 3 : i32} : () -> memref<512x3x3x4xf16>
    %4 = "okm.arg_to_memref"() {index = 4 : i32} : () -> memref<512xf16>
    %5 = "okm.arg_to_memref"() {index = 5 : i32} : () -> memref<512xf16>
    %6 = "okm.arg_to_memref"() {index = 6 : i32} : () -> memref<512xf16>
    %7 = "okm.arg_to_memref"() {index = 7 : i32} : () -> memref<512x3x3x512xf16>
    %8 = "okm.arg_to_memref"() {index = 8 : i32} : () -> memref<512xf16>
    %9 = "okm.arg_to_memref"() {index = 9 : i32} : () -> memref<512xf16>
    %10 = "okm.arg_to_memref"() {index = 10 : i32} : () -> memref<512xf16>
    %11 = "okm.arg_to_memref"() {index = 11 : i32} : () -> memref<512x3x3x512xf16>
    %12 = "okm.arg_to_memref"() {index = 12 : i32} : () -> memref<512xf16>
    %13 = "okm.arg_to_memref"() {index = 13 : i32} : () -> memref<512xf16>
    %14 = "okm.arg_to_memref"() {index = 14 : i32} : () -> memref<512xf16>
    %15 = "okm.arg_to_memref"() {index = 15 : i32} : () -> memref<512x512xf16>
    %16 = "okm.arg_to_memref"() {index = 16 : i32} : () -> memref<512x512xf16>
    %17 = "okm.arg_to_memref"() {index = 17 : i32} : () -> memref<512x512xf16>
    %18 = "okm.arg_to_memref"() {index = 18 : i32} : () -> memref<512xf16>
    %19 = "okm.arg_to_memref"() {index = 19 : i32} : () -> memref<512xf16>
    %20 = "okm.arg_to_memref"() {index = 20 : i32} : () -> memref<512xf16>
    %21 = "okm.arg_to_memref"() {index = 21 : i32} : () -> memref<512x512xf16>
    %22 = "okm.arg_to_memref"() {index = 22 : i32} : () -> memref<512xf16>
    %23 = "okm.arg_to_memref"() {index = 23 : i32} : () -> memref<512xf16>
    %24 = "okm.arg_to_memref"() {index = 24 : i32} : () -> memref<512xf16>
    %25 = "okm.arg_to_memref"() {index = 25 : i32} : () -> memref<512x3x3x512xf16>
    %26 = "okm.arg_to_memref"() {index = 26 : i32} : () -> memref<512xf16>
    %27 = "okm.arg_to_memref"() {index = 27 : i32} : () -> memref<512xf16>
    %28 = "okm.arg_to_memref"() {index = 28 : i32} : () -> memref<512xf16>
    %29 = "okm.arg_to_memref"() {index = 29 : i32} : () -> memref<512x3x3x512xf16>
    %30 = "okm.arg_to_memref"() {index = 30 : i32} : () -> memref<512xf16>
    %31 = "okm.arg_to_memref"() {index = 31 : i32} : () -> memref<512xf16>
    %32 = "okm.arg_to_memref"() {index = 32 : i32} : () -> memref<512xf16>
    %33 = "okm.arg_to_memref"() {index = 33 : i32} : () -> memref<512x3x3x512xf16>
    %34 = "okm.arg_to_memref"() {index = 34 : i32} : () -> memref<512xf16>
    %35 = "okm.arg_to_memref"() {index = 35 : i32} : () -> memref<512xf16>
    %36 = "okm.arg_to_memref"() {index = 36 : i32} : () -> memref<512xf16>
    %37 = "okm.arg_to_memref"() {index = 37 : i32} : () -> memref<512x3x3x512xf16>
    %38 = "okm.arg_to_memref"() {index = 38 : i32} : () -> memref<512xf16>
    %39 = "okm.arg_to_memref"() {index = 39 : i32} : () -> memref<512xf16>
    %40 = "okm.arg_to_memref"() {index = 40 : i32} : () -> memref<512xf16>
    %41 = "okm.arg_to_memref"() {index = 41 : i32} : () -> memref<512x3x3x512xf16>
    %42 = "okm.arg_to_memref"() {index = 42 : i32} : () -> memref<512xf16>
    %43 = "okm.arg_to_memref"() {index = 43 : i32} : () -> memref<512xf16>
    %44 = "okm.arg_to_memref"() {index = 44 : i32} : () -> memref<512xf16>
    %45 = "okm.arg_to_memref"() {index = 45 : i32} : () -> memref<512x3x3x512xf16>
    %46 = "okm.arg_to_memref"() {index = 46 : i32} : () -> memref<512xf16>
    %47 = "okm.arg_to_memref"() {index = 47 : i32} : () -> memref<512xf16>
    %48 = "okm.arg_to_memref"() {index = 48 : i32} : () -> memref<512xf16>
    %49 = "okm.arg_to_memref"() {index = 49 : i32} : () -> memref<512x3x3x512xf16>
    %50 = "okm.arg_to_memref"() {index = 50 : i32} : () -> memref<512xf16>
    %51 = "okm.arg_to_memref"() {index = 51 : i32} : () -> memref<512xf16>
    %52 = "okm.arg_to_memref"() {index = 52 : i32} : () -> memref<512xf16>
    %53 = "okm.arg_to_memref"() {index = 53 : i32} : () -> memref<512x3x3x512xf16>
    %54 = "okm.arg_to_memref"() {index = 54 : i32} : () -> memref<512xf16>
    %55 = "okm.arg_to_memref"() {index = 55 : i32} : () -> memref<512x3x3x512xf16>
    %56 = "okm.arg_to_memref"() {index = 56 : i32} : () -> memref<512xf16>
    %57 = "okm.arg_to_memref"() {index = 57 : i32} : () -> memref<512xf16>
    %58 = "okm.arg_to_memref"() {index = 58 : i32} : () -> memref<512xf16>
    %59 = "okm.arg_to_memref"() {index = 59 : i32} : () -> memref<512x3x3x512xf16>
    %60 = "okm.arg_to_memref"() {index = 60 : i32} : () -> memref<512xf16>
    %61 = "okm.arg_to_memref"() {index = 61 : i32} : () -> memref<512xf16>
    %62 = "okm.arg_to_memref"() {index = 62 : i32} : () -> memref<512xf16>
    %63 = "okm.arg_to_memref"() {index = 63 : i32} : () -> memref<512x3x3x512xf16>
    %64 = "okm.arg_to_memref"() {index = 64 : i32} : () -> memref<512xf16>
    %65 = "okm.arg_to_memref"() {index = 65 : i32} : () -> memref<512xf16>
    %66 = "okm.arg_to_memref"() {index = 66 : i32} : () -> memref<512xf16>
    %67 = "okm.arg_to_memref"() {index = 67 : i32} : () -> memref<512x3x3x512xf16>
    %68 = "okm.arg_to_memref"() {index = 68 : i32} : () -> memref<512xf16>
    %69 = "okm.arg_to_memref"() {index = 69 : i32} : () -> memref<512xf16>
    %70 = "okm.arg_to_memref"() {index = 70 : i32} : () -> memref<512xf16>
    %71 = "okm.arg_to_memref"() {index = 71 : i32} : () -> memref<512x3x3x512xf16>
    %72 = "okm.arg_to_memref"() {index = 72 : i32} : () -> memref<512xf16>
    %73 = "okm.arg_to_memref"() {index = 73 : i32} : () -> memref<512xf16>
    %74 = "okm.arg_to_memref"() {index = 74 : i32} : () -> memref<512xf16>
    %75 = "okm.arg_to_memref"() {index = 75 : i32} : () -> memref<512x3x3x512xf16>
    %76 = "okm.arg_to_memref"() {index = 76 : i32} : () -> memref<512xf16>
    %77 = "okm.arg_to_memref"() {index = 77 : i32} : () -> memref<512xf16>
    %78 = "okm.arg_to_memref"() {index = 78 : i32} : () -> memref<512xf16>
    %79 = "okm.arg_to_memref"() {index = 79 : i32} : () -> memref<512x3x3x512xf16>
    %80 = "okm.arg_to_memref"() {index = 80 : i32} : () -> memref<512xf16>
    %81 = "okm.arg_to_memref"() {index = 81 : i32} : () -> memref<512x3x3x512xf16>
    %82 = "okm.arg_to_memref"() {index = 82 : i32} : () -> memref<512xf16>
    %83 = "okm.arg_to_memref"() {index = 83 : i32} : () -> memref<512xf16>
    %84 = "okm.arg_to_memref"() {index = 84 : i32} : () -> memref<512xf16>
    %85 = "okm.arg_to_memref"() {index = 85 : i32} : () -> memref<256x1x1x512xf16>
    %86 = "okm.arg_to_memref"() {index = 86 : i32} : () -> memref<256xf16>
    %87 = "okm.arg_to_memref"() {index = 87 : i32} : () -> memref<256x3x3x512xf16>
    %88 = "okm.arg_to_memref"() {index = 88 : i32} : () -> memref<256xf16>
    %89 = "okm.arg_to_memref"() {index = 89 : i32} : () -> memref<256xf16>
    %90 = "okm.arg_to_memref"() {index = 90 : i32} : () -> memref<256xf16>
    %91 = "okm.arg_to_memref"() {index = 91 : i32} : () -> memref<256x3x3x256xf16>
    %92 = "okm.arg_to_memref"() {index = 92 : i32} : () -> memref<256xf16>
    %93 = "okm.arg_to_memref"() {index = 93 : i32} : () -> memref<256xf16>
    %94 = "okm.arg_to_memref"() {index = 94 : i32} : () -> memref<256xf16>
    %95 = "okm.arg_to_memref"() {index = 95 : i32} : () -> memref<256x3x3x256xf16>
    %96 = "okm.arg_to_memref"() {index = 96 : i32} : () -> memref<256xf16>
    %97 = "okm.arg_to_memref"() {index = 97 : i32} : () -> memref<256xf16>
    %98 = "okm.arg_to_memref"() {index = 98 : i32} : () -> memref<256xf16>
    %99 = "okm.arg_to_memref"() {index = 99 : i32} : () -> memref<256x3x3x256xf16>
    %100 = "okm.arg_to_memref"() {index = 100 : i32} : () -> memref<256xf16>
    %101 = "okm.arg_to_memref"() {index = 101 : i32} : () -> memref<256xf16>
    %102 = "okm.arg_to_memref"() {index = 102 : i32} : () -> memref<256xf16>
    %103 = "okm.arg_to_memref"() {index = 103 : i32} : () -> memref<256x3x3x256xf16>
    %104 = "okm.arg_to_memref"() {index = 104 : i32} : () -> memref<256xf16>
    %105 = "okm.arg_to_memref"() {index = 105 : i32} : () -> memref<256xf16>
    %106 = "okm.arg_to_memref"() {index = 106 : i32} : () -> memref<256xf16>
    %107 = "okm.arg_to_memref"() {index = 107 : i32} : () -> memref<256x3x3x256xf16>
    %108 = "okm.arg_to_memref"() {index = 108 : i32} : () -> memref<256xf16>
    %109 = "okm.arg_to_memref"() {index = 109 : i32} : () -> memref<256x3x3x256xf16>
    %110 = "okm.arg_to_memref"() {index = 110 : i32} : () -> memref<256xf16>
    %111 = "okm.arg_to_memref"() {index = 111 : i32} : () -> memref<256xf16>
    %112 = "okm.arg_to_memref"() {index = 112 : i32} : () -> memref<256xf16>
    %113 = "okm.arg_to_memref"() {index = 113 : i32} : () -> memref<128x1x1x256xf16>
    %114 = "okm.arg_to_memref"() {index = 114 : i32} : () -> memref<128xf16>
    %115 = "okm.arg_to_memref"() {index = 115 : i32} : () -> memref<128x3x3x256xf16>
    %116 = "okm.arg_to_memref"() {index = 116 : i32} : () -> memref<128xf16>
    %117 = "okm.arg_to_memref"() {index = 117 : i32} : () -> memref<128xf16>
    %118 = "okm.arg_to_memref"() {index = 118 : i32} : () -> memref<128xf16>
    %119 = "okm.arg_to_memref"() {index = 119 : i32} : () -> memref<128x3x3x128xf16>
    %120 = "okm.arg_to_memref"() {index = 120 : i32} : () -> memref<128xf16>
    %121 = "okm.arg_to_memref"() {index = 121 : i32} : () -> memref<128xf16>
    %122 = "okm.arg_to_memref"() {index = 122 : i32} : () -> memref<128xf16>
    %123 = "okm.arg_to_memref"() {index = 123 : i32} : () -> memref<128x3x3x128xf16>
    %124 = "okm.arg_to_memref"() {index = 124 : i32} : () -> memref<128xf16>
    %125 = "okm.arg_to_memref"() {index = 125 : i32} : () -> memref<128xf16>
    %126 = "okm.arg_to_memref"() {index = 126 : i32} : () -> memref<128xf16>
    %127 = "okm.arg_to_memref"() {index = 127 : i32} : () -> memref<128x3x3x128xf16>
    %128 = "okm.arg_to_memref"() {index = 128 : i32} : () -> memref<128xf16>
    %129 = "okm.arg_to_memref"() {index = 129 : i32} : () -> memref<128xf16>
    %130 = "okm.arg_to_memref"() {index = 130 : i32} : () -> memref<128xf16>
    %131 = "okm.arg_to_memref"() {index = 131 : i32} : () -> memref<128x3x3x128xf16>
    %132 = "okm.arg_to_memref"() {index = 132 : i32} : () -> memref<128xf16>
    %133 = "okm.arg_to_memref"() {index = 133 : i32} : () -> memref<128xf16>
    %134 = "okm.arg_to_memref"() {index = 134 : i32} : () -> memref<128xf16>
    %135 = "okm.arg_to_memref"() {index = 135 : i32} : () -> memref<128x3x3x128xf16>
    %136 = "okm.arg_to_memref"() {index = 136 : i32} : () -> memref<128xf16>
    %137 = "okm.arg_to_memref"() {index = 137 : i32} : () -> memref<128xf16>
    %138 = "okm.arg_to_memref"() {index = 138 : i32} : () -> memref<128xf16>
    %139 = "okm.arg_to_memref"() {index = 139 : i32} : () -> memref<3x3x3x128xf16>
    %140 = "okm.arg_to_memref"() {index = 140 : i32} : () -> memref<3xf16>
    %141 = "okm.plan_memref"() : () -> memref<1x4x64x64xf16>
    %142 = "okm.wrapper_kernel"(%0, %141) ({
      %486 = bufferization.to_tensor %0 : memref<1x4x64x64xf16>
      %487 = "oneflow.scalar_mul"(%486) {device_name = ["@0:0"], device_tag = "cuda", float_operand = 5.4899807850672522 : f64, has_float_operand = true, has_int_operand = false, hierarchy = [1], int_operand = 0 : si64, op_name = "vae_post_process-scalar_mul-0", scope_symbol_id = 16 : i64} : (tensor<1x4x64x64xf16>) -> tensor<1x4x64x64xf16>
      %488 = bufferization.to_memref %487 : memref<1x4x64x64xf16>
      okm.return %488 : memref<1x4x64x64xf16>
    }) : (memref<1x4x64x64xf16>, memref<1x4x64x64xf16>) -> memref<1x4x64x64xf16>
    %143 = "okm.plan_memref"() : () -> memref<1x64x64x4xf16>
    %144 = "okm.wrapper_kernel"(%142, %143) ({
      %486 = bufferization.to_tensor %142 : memref<1x4x64x64xf16>
      %487 = "oneflow.transpose"(%486) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.post_quant_conv-conv2d-1_transpose_input_0", perm = [0 : si32, 2 : si32, 3 : si32, 1 : si32], scope_symbol_id = 28 : i64} : (tensor<1x4x64x64xf16>) -> tensor<1x64x64x4xf16>
      %488 = bufferization.to_memref %487 : memref<1x64x64x4xf16>
      okm.return %488 : memref<1x64x64x4xf16>
    }) : (memref<1x4x64x64xf16>, memref<1x64x64x4xf16>) -> memref<1x64x64x4xf16>
    %145 = "okm.plan_memref"() : () -> memref<1x64x64x4xf16>
    %146 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %147 = "okm.wrapper_kernel"(%144, %1, %2, %145, %146) ({
      %486 = bufferization.to_tensor %144 : memref<1x64x64x4xf16>
      %487 = bufferization.to_tensor %1 : memref<4x1x1x4xf16>
      %488 = bufferization.to_tensor %2 : memref<4xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 4 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [1 : si32, 1 : si32], op_name = "vae_post_process.vae.post_quant_conv-conv2d-1", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [0 : si32, 0 : si32], scope_symbol_id = 28 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x4xf16>, tensor<4x1x1x4xf16>, tensor<4xf16>) -> tensor<1x64x64x4xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x4xf16>
      okm.return %490 : memref<1x64x64x4xf16>
    }) : (memref<1x64x64x4xf16>, memref<4x1x1x4xf16>, memref<4xf16>, memref<1x64x64x4xf16>, memref<134217728xi8>) -> memref<1x64x64x4xf16>
    %148 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %149 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %150 = "okm.wrapper_kernel"(%147, %3, %4, %148, %149) ({
      %486 = bufferization.to_tensor %147 : memref<1x64x64x4xf16>
      %487 = bufferization.to_tensor %3 : memref<512x3x3x4xf16>
      %488 = bufferization.to_tensor %4 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.conv_in-conv2d-2", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 41 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x4xf16>, tensor<512x3x3x4xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x4xf16>, memref<512x3x3x4xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %151 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %152 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %153 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %154:3 = "okm.wrapper_kernel"(%150, %5, %6, %151, %152, %153) ({
      %486 = bufferization.to_tensor %150 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %5 : memref<512xf16>
      %488 = bufferization.to_tensor %6 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.mid_block.resnets.0.norm1-group_norm-3_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 55 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %155 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %156 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %157 = "okm.wrapper_kernel"(%154#0, %7, %8, %155, %156) ({
      %486 = bufferization.to_tensor %154#0 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %7 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %8 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.mid_block.resnets.0.conv1-conv2d-5", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 71 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %158 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %159 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %160 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %161:3 = "okm.wrapper_kernel"(%157, %9, %10, %158, %159, %160) ({
      %486 = bufferization.to_tensor %157 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %9 : memref<512xf16>
      %488 = bufferization.to_tensor %10 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.mid_block.resnets.0.norm2-group_norm-6_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 83 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %162 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %163 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %164 = "okm.wrapper_kernel"(%161#0, %11, %12, %162, %163) ({
      %486 = bufferization.to_tensor %161#0 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %11 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %12 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.mid_block.resnets.0.conv2-conv2d-8", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 99 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %165 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %166 = "okm.wrapper_kernel"(%150, %164, %165) ({
      %486 = bufferization.to_tensor %150 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %164 : memref<1x64x64x512xf16>
      %488 = "oneflow.add_n2"(%486, %487) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.resnets.0-add_n-9", op_type_name = "add_n", scope_symbol_id = 102 : i64} : (tensor<1x64x64x512xf16>, tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %489 = bufferization.to_memref %488 : memref<1x64x64x512xf16>
      okm.return %489 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %167 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %168 = "okm.wrapper_kernel"(%166, %167) ({
      %486 = bufferization.to_tensor %166 : memref<1x64x64x512xf16>
      %487 = "oneflow.scalar_div"(%486) {device_name = ["@0:0"], device_tag = "cuda", float_operand = 0.000000e+00 : f64, has_float_operand = false, has_int_operand = true, hierarchy = [1], int_operand = 1 : si64, op_name = "vae_post_process.vae.decoder.mid_block.resnets.0-scalar_div-10", scope_symbol_id = 102 : i64} : (tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %488 = bufferization.to_memref %487 : memref<1x64x64x512xf16>
      okm.return %488 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %169 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %170 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %171 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %172:3 = "okm.wrapper_kernel"(%168, %13, %14, %169, %170, %171) ({
      %486 = bufferization.to_tensor %168 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %13 : memref<512xf16>
      %488 = bufferization.to_tensor %14 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "none", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.mid_block.attentions.0.group_norm-group_norm-11", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 118 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %173 = "okm.plan_memref"() : () -> memref<1x512x64x64xf16>
    %174 = "okm.wrapper_kernel"(%172#0, %173) ({
      %486 = bufferization.to_tensor %172#0 : memref<1x64x64x512xf16>
      %487 = "oneflow.transpose"(%486) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.attentions.0.group_norm-group_norm-11_transpose_output_0", perm = [0 : si32, 3 : si32, 1 : si32, 2 : si32], scope_symbol_id = 118 : i64} : (tensor<1x64x64x512xf16>) -> tensor<1x512x64x64xf16>
      %488 = bufferization.to_memref %487 : memref<1x512x64x64xf16>
      okm.return %488 : memref<1x512x64x64xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x512x64x64xf16>) -> memref<1x512x64x64xf16>
    %175 = "okm.plan_memref"() : () -> memref<1x512x4096xf16>
    %176 = "okm.wrapper_kernel"(%174, %175) ({
      %486 = bufferization.to_tensor %174 : memref<1x512x64x64xf16>
      %487 = "oneflow.reshape"(%486) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.attentions.0-reshape-12", scope_symbol_id = 121 : i64, shape = [1 : si64, 512 : si64, 4096 : si64]} : (tensor<1x512x64x64xf16>) -> tensor<1x512x4096xf16>
      %488 = bufferization.to_memref %487 : memref<1x512x4096xf16>
      okm.return %488 : memref<1x512x4096xf16>
    }) : (memref<1x512x64x64xf16>, memref<1x512x4096xf16>) -> memref<1x512x4096xf16>
    %177 = "okm.plan_memref"() : () -> memref<1x4096x512xf16>
    %178 = "okm.wrapper_kernel"(%176, %177) ({
      %486 = bufferization.to_tensor %176 : memref<1x512x4096xf16>
      %487 = "oneflow.transpose"(%486) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.attentions.0-transpose-13", perm = [0 : si32, 2 : si32, 1 : si32], scope_symbol_id = 121 : i64} : (tensor<1x512x4096xf16>) -> tensor<1x4096x512xf16>
      %488 = bufferization.to_memref %487 : memref<1x4096x512xf16>
      okm.return %488 : memref<1x4096x512xf16>
    }) : (memref<1x512x4096xf16>, memref<1x4096x512xf16>) -> memref<1x4096x512xf16>
    %179 = "okm.plan_memref"() : () -> memref<1x4096x512xf16>
    %180 = "okm.plan_memref"() : () -> memref<1x4096x512xf16>
    %181 = "okm.plan_memref"() : () -> memref<1x4096x512xf16>
    %182 = "okm.plan_memref"() : () -> memref<1536xi8>
    %183:3 = "okm.wrapper_kernel"(%178, %178, %178, %15, %16, %17, %18, %19, %20, %179, %180, %181, %182) ({
      %486 = bufferization.to_tensor %178 : memref<1x4096x512xf16>
      %487 = bufferization.to_tensor %15 : memref<512x512xf16>
      %488 = bufferization.to_tensor %16 : memref<512x512xf16>
      %489 = bufferization.to_tensor %17 : memref<512x512xf16>
      %490 = bufferization.to_tensor %18 : memref<512xf16>
      %491 = bufferization.to_tensor %19 : memref<512xf16>
      %492 = bufferization.to_tensor %20 : memref<512xf16>
      %493:3 = "oneflow.grouped_matmul_bias"(%486, %486, %486, %487, %488, %489, %490, %491, %492) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "grouped_matmul_vae_post_process.vae.decoder.mid_block.attentions.0.value-broadcast_matmul-18", operand_segment_sizes = dense<3> : vector<3xi32>, scope_symbol_id = 162 : i64} : (tensor<1x4096x512xf16>, tensor<1x4096x512xf16>, tensor<1x4096x512xf16>, tensor<512x512xf16>, tensor<512x512xf16>, tensor<512x512xf16>, tensor<512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x4096x512xf16>, tensor<1x4096x512xf16>, tensor<1x4096x512xf16>)
      %494 = bufferization.to_memref %493#0 : memref<1x4096x512xf16>
      %495 = bufferization.to_memref %493#1 : memref<1x4096x512xf16>
      %496 = bufferization.to_memref %493#2 : memref<1x4096x512xf16>
      okm.return %494, %495, %496 : memref<1x4096x512xf16>, memref<1x4096x512xf16>, memref<1x4096x512xf16>
    }) : (memref<1x4096x512xf16>, memref<1x4096x512xf16>, memref<1x4096x512xf16>, memref<512x512xf16>, memref<512x512xf16>, memref<512x512xf16>, memref<512xf16>, memref<512xf16>, memref<512xf16>, memref<1x4096x512xf16>, memref<1x4096x512xf16>, memref<1x4096x512xf16>, memref<1536xi8>) -> (memref<1x4096x512xf16>, memref<1x4096x512xf16>, memref<1x4096x512xf16>)
    %184 = "okm.plan_memref"() : () -> memref<1x4096x512xf16>
    %185 = "okm.plan_memref"() : () -> memref<20972032xi8>
    %186 = "okm.wrapper_kernel"(%183#2, %183#1, %183#0, %184, %185) ({
      %486 = bufferization.to_tensor %183#2 : memref<1x4096x512xf16>
      %487 = bufferization.to_tensor %183#1 : memref<1x4096x512xf16>
      %488 = bufferization.to_tensor %183#0 : memref<1x4096x512xf16>
      %489 = "oneflow.fused_multi_head_attention_inference"(%486, %487, %488) {causal = false, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], key_hidden_slice_end = -1 : si64, key_hidden_slice_start = 0 : si64, num_heads = 1 : si64, op_name = "vae_post_process.vae.decoder.mid_block.attentions.0-fused_multi_head_attention_inference-20", query_hidden_slice_end = -1 : si64, query_hidden_slice_start = 0 : si64, scope_symbol_id = 121 : i64, value_hidden_slice_end = -1 : si64, value_hidden_slice_start = 0 : si64} : (tensor<1x4096x512xf16>, tensor<1x4096x512xf16>, tensor<1x4096x512xf16>) -> tensor<1x4096x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x4096x512xf16>
      okm.return %490 : memref<1x4096x512xf16>
    }) : (memref<1x4096x512xf16>, memref<1x4096x512xf16>, memref<1x4096x512xf16>, memref<1x4096x512xf16>, memref<20972032xi8>) -> memref<1x4096x512xf16>
    %187 = "okm.plan_memref"() : () -> memref<1x4096x512xf16>
    %188 = "okm.wrapper_kernel"(%186, %21, %187) ({
      %486 = bufferization.to_tensor %186 : memref<1x4096x512xf16>
      %487 = bufferization.to_tensor %21 : memref<512x512xf16>
      %488 = "oneflow.broadcast_matmul"(%486, %487) {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.attentions.0.proj_attn-broadcast_matmul-21", scope_symbol_id = 180 : i64, transpose_a = false, transpose_b = true} : (tensor<1x4096x512xf16>, tensor<512x512xf16>) -> tensor<1x4096x512xf16>
      %489 = bufferization.to_memref %488 : memref<1x4096x512xf16>
      okm.return %489 : memref<1x4096x512xf16>
    }) : (memref<1x4096x512xf16>, memref<512x512xf16>, memref<1x4096x512xf16>) -> memref<1x4096x512xf16>
    %189 = "okm.plan_memref"() : () -> memref<1x4096x512xf16>
    %190 = "okm.wrapper_kernel"(%188, %22, %189) ({
      %486 = bufferization.to_tensor %188 : memref<1x4096x512xf16>
      %487 = bufferization.to_tensor %22 : memref<512xf16>
      %488 = "oneflow.broadcast_add"(%486, %487) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.attentions.0.proj_attn-broadcast_add-22", scope_symbol_id = 180 : i64} : (tensor<1x4096x512xf16>, tensor<512xf16>) -> tensor<1x4096x512xf16>
      %489 = bufferization.to_memref %488 : memref<1x4096x512xf16>
      okm.return %489 : memref<1x4096x512xf16>
    }) : (memref<1x4096x512xf16>, memref<512xf16>, memref<1x4096x512xf16>) -> memref<1x4096x512xf16>
    %191 = "okm.plan_memref"() : () -> memref<1x512x4096xf16>
    %192 = "okm.wrapper_kernel"(%190, %191) ({
      %486 = bufferization.to_tensor %190 : memref<1x4096x512xf16>
      %487 = "oneflow.transpose"(%486) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.attentions.0-transpose-23", perm = [0 : si32, 2 : si32, 1 : si32], scope_symbol_id = 121 : i64} : (tensor<1x4096x512xf16>) -> tensor<1x512x4096xf16>
      %488 = bufferization.to_memref %487 : memref<1x512x4096xf16>
      okm.return %488 : memref<1x512x4096xf16>
    }) : (memref<1x4096x512xf16>, memref<1x512x4096xf16>) -> memref<1x512x4096xf16>
    %193 = "okm.plan_memref"() : () -> memref<1x512x64x64xf16>
    %194 = "okm.wrapper_kernel"(%192, %193) ({
      %486 = bufferization.to_tensor %192 : memref<1x512x4096xf16>
      %487 = "oneflow.reshape"(%486) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.attentions.0-reshape-24", scope_symbol_id = 121 : i64, shape = [1 : si64, 512 : si64, 64 : si64, 64 : si64]} : (tensor<1x512x4096xf16>) -> tensor<1x512x64x64xf16>
      %488 = bufferization.to_memref %487 : memref<1x512x64x64xf16>
      okm.return %488 : memref<1x512x64x64xf16>
    }) : (memref<1x512x4096xf16>, memref<1x512x64x64xf16>) -> memref<1x512x64x64xf16>
    %195 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %196 = "okm.wrapper_kernel"(%194, %195) ({
      %486 = bufferization.to_tensor %194 : memref<1x512x64x64xf16>
      %487 = "oneflow.transpose"(%486) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.attentions.0-add_n-25_transpose_input_0", perm = [0 : si32, 2 : si32, 3 : si32, 1 : si32], scope_symbol_id = 121 : i64} : (tensor<1x512x64x64xf16>) -> tensor<1x64x64x512xf16>
      %488 = bufferization.to_memref %487 : memref<1x64x64x512xf16>
      okm.return %488 : memref<1x64x64x512xf16>
    }) : (memref<1x512x64x64xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %197 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %198 = "okm.wrapper_kernel"(%196, %168, %197) ({
      %486 = bufferization.to_tensor %196 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %168 : memref<1x64x64x512xf16>
      %488 = "oneflow.add_n2"(%486, %487) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.attentions.0-add_n-25", op_type_name = "add_n", scope_symbol_id = 121 : i64} : (tensor<1x64x64x512xf16>, tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %489 = bufferization.to_memref %488 : memref<1x64x64x512xf16>
      okm.return %489 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %199 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %200 = "okm.wrapper_kernel"(%198, %199) ({
      %486 = bufferization.to_tensor %198 : memref<1x64x64x512xf16>
      %487 = "oneflow.scalar_div"(%486) {device_name = ["@0:0"], device_tag = "cuda", float_operand = 0.000000e+00 : f64, has_float_operand = false, has_int_operand = true, hierarchy = [1], int_operand = 1 : si64, op_name = "vae_post_process.vae.decoder.mid_block.attentions.0-scalar_div-26", scope_symbol_id = 121 : i64} : (tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %488 = bufferization.to_memref %487 : memref<1x64x64x512xf16>
      okm.return %488 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %201 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %202 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %203 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %204:3 = "okm.wrapper_kernel"(%200, %23, %24, %201, %202, %203) ({
      %486 = bufferization.to_tensor %200 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %23 : memref<512xf16>
      %488 = bufferization.to_tensor %24 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.mid_block.resnets.1.norm1-group_norm-27_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 212 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %205 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %206 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %207 = "okm.wrapper_kernel"(%204#0, %25, %26, %205, %206) ({
      %486 = bufferization.to_tensor %204#0 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %25 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %26 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.mid_block.resnets.1.conv1-conv2d-29", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 228 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %208 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %209 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %210 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %211:3 = "okm.wrapper_kernel"(%207, %27, %28, %208, %209, %210) ({
      %486 = bufferization.to_tensor %207 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %27 : memref<512xf16>
      %488 = bufferization.to_tensor %28 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.mid_block.resnets.1.norm2-group_norm-30_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 240 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %212 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %213 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %214 = "okm.wrapper_kernel"(%211#0, %29, %30, %212, %213) ({
      %486 = bufferization.to_tensor %211#0 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %29 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %30 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.mid_block.resnets.1.conv2-conv2d-32", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 256 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %215 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %216 = "okm.wrapper_kernel"(%200, %214, %215) ({
      %486 = bufferization.to_tensor %200 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %214 : memref<1x64x64x512xf16>
      %488 = "oneflow.add_n2"(%486, %487) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.mid_block.resnets.1-add_n-33", op_type_name = "add_n", scope_symbol_id = 259 : i64} : (tensor<1x64x64x512xf16>, tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %489 = bufferization.to_memref %488 : memref<1x64x64x512xf16>
      okm.return %489 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %217 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %218 = "okm.wrapper_kernel"(%216, %217) ({
      %486 = bufferization.to_tensor %216 : memref<1x64x64x512xf16>
      %487 = "oneflow.scalar_div"(%486) {device_name = ["@0:0"], device_tag = "cuda", float_operand = 0.000000e+00 : f64, has_float_operand = false, has_int_operand = true, hierarchy = [1], int_operand = 1 : si64, op_name = "vae_post_process.vae.decoder.mid_block.resnets.1-scalar_div-34", scope_symbol_id = 259 : i64} : (tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %488 = bufferization.to_memref %487 : memref<1x64x64x512xf16>
      okm.return %488 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %219 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %220 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %221 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %222:3 = "okm.wrapper_kernel"(%218, %31, %32, %219, %220, %221) ({
      %486 = bufferization.to_tensor %218 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %31 : memref<512xf16>
      %488 = bufferization.to_tensor %32 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.0.norm1-group_norm-35_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 276 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %223 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %224 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %225 = "okm.wrapper_kernel"(%222#0, %33, %34, %223, %224) ({
      %486 = bufferization.to_tensor %222#0 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %33 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %34 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.0.conv1-conv2d-37", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 292 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %226 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %227 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %228 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %229:3 = "okm.wrapper_kernel"(%225, %35, %36, %226, %227, %228) ({
      %486 = bufferization.to_tensor %225 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %35 : memref<512xf16>
      %488 = bufferization.to_tensor %36 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.0.norm2-group_norm-38_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 304 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %230 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %231 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %232 = "okm.wrapper_kernel"(%229#0, %37, %38, %230, %231) ({
      %486 = bufferization.to_tensor %229#0 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %37 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %38 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.0.conv2-conv2d-40", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 320 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %233 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %234 = "okm.wrapper_kernel"(%218, %232, %233) ({
      %486 = bufferization.to_tensor %218 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %232 : memref<1x64x64x512xf16>
      %488 = "oneflow.add_n2"(%486, %487) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.0-add_n-41", op_type_name = "add_n", scope_symbol_id = 323 : i64} : (tensor<1x64x64x512xf16>, tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %489 = bufferization.to_memref %488 : memref<1x64x64x512xf16>
      okm.return %489 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %235 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %236 = "okm.wrapper_kernel"(%234, %235) ({
      %486 = bufferization.to_tensor %234 : memref<1x64x64x512xf16>
      %487 = "oneflow.scalar_div"(%486) {device_name = ["@0:0"], device_tag = "cuda", float_operand = 1.000000e+00 : f64, has_float_operand = true, has_int_operand = false, hierarchy = [1], int_operand = 0 : si64, op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.0-scalar_div-42", scope_symbol_id = 323 : i64} : (tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %488 = bufferization.to_memref %487 : memref<1x64x64x512xf16>
      okm.return %488 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %237 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %238 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %239 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %240:3 = "okm.wrapper_kernel"(%236, %39, %40, %237, %238, %239) ({
      %486 = bufferization.to_tensor %236 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %39 : memref<512xf16>
      %488 = bufferization.to_tensor %40 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.1.norm1-group_norm-43_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 339 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %241 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %242 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %243 = "okm.wrapper_kernel"(%240#0, %41, %42, %241, %242) ({
      %486 = bufferization.to_tensor %240#0 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %41 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %42 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.1.conv1-conv2d-45", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 355 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %244 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %245 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %246 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %247:3 = "okm.wrapper_kernel"(%243, %43, %44, %244, %245, %246) ({
      %486 = bufferization.to_tensor %243 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %43 : memref<512xf16>
      %488 = bufferization.to_tensor %44 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.1.norm2-group_norm-46_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 367 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %248 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %249 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %250 = "okm.wrapper_kernel"(%247#0, %45, %46, %248, %249) ({
      %486 = bufferization.to_tensor %247#0 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %45 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %46 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.1.conv2-conv2d-48", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 383 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %251 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %252 = "okm.wrapper_kernel"(%236, %250, %251) ({
      %486 = bufferization.to_tensor %236 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %250 : memref<1x64x64x512xf16>
      %488 = "oneflow.add_n2"(%486, %487) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.1-add_n-49", op_type_name = "add_n", scope_symbol_id = 386 : i64} : (tensor<1x64x64x512xf16>, tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %489 = bufferization.to_memref %488 : memref<1x64x64x512xf16>
      okm.return %489 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %253 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %254 = "okm.wrapper_kernel"(%252, %253) ({
      %486 = bufferization.to_tensor %252 : memref<1x64x64x512xf16>
      %487 = "oneflow.scalar_div"(%486) {device_name = ["@0:0"], device_tag = "cuda", float_operand = 1.000000e+00 : f64, has_float_operand = true, has_int_operand = false, hierarchy = [1], int_operand = 0 : si64, op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.1-scalar_div-50", scope_symbol_id = 386 : i64} : (tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %488 = bufferization.to_memref %487 : memref<1x64x64x512xf16>
      okm.return %488 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %255 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %256 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %257 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %258:3 = "okm.wrapper_kernel"(%254, %47, %48, %255, %256, %257) ({
      %486 = bufferization.to_tensor %254 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %47 : memref<512xf16>
      %488 = bufferization.to_tensor %48 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.2.norm1-group_norm-51_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 402 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %259 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %260 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %261 = "okm.wrapper_kernel"(%258#0, %49, %50, %259, %260) ({
      %486 = bufferization.to_tensor %258#0 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %49 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %50 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.2.conv1-conv2d-53", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 418 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %262 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %263 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %264 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %265:3 = "okm.wrapper_kernel"(%261, %51, %52, %262, %263, %264) ({
      %486 = bufferization.to_tensor %261 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %51 : memref<512xf16>
      %488 = bufferization.to_tensor %52 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.2.norm2-group_norm-54_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 430 : i64} : (tensor<1x64x64x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x64x64x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x64x64x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x64x64x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x64x64x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %266 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %267 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %268 = "okm.wrapper_kernel"(%265#0, %53, %54, %266, %267) ({
      %486 = bufferization.to_tensor %265#0 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %53 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %54 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.2.conv2-conv2d-56", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 446 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x64x64x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x64x64x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x64x64x512xf16>
      okm.return %490 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x64x64x512xf16>, memref<134217728xi8>) -> memref<1x64x64x512xf16>
    %269 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %270 = "okm.wrapper_kernel"(%254, %268, %269) ({
      %486 = bufferization.to_tensor %254 : memref<1x64x64x512xf16>
      %487 = bufferization.to_tensor %268 : memref<1x64x64x512xf16>
      %488 = "oneflow.add_n2"(%486, %487) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.2-add_n-57", op_type_name = "add_n", scope_symbol_id = 449 : i64} : (tensor<1x64x64x512xf16>, tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %489 = bufferization.to_memref %488 : memref<1x64x64x512xf16>
      okm.return %489 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %271 = "okm.plan_memref"() : () -> memref<1x64x64x512xf16>
    %272 = "okm.wrapper_kernel"(%270, %271) ({
      %486 = bufferization.to_tensor %270 : memref<1x64x64x512xf16>
      %487 = "oneflow.scalar_div"(%486) {device_name = ["@0:0"], device_tag = "cuda", float_operand = 1.000000e+00 : f64, has_float_operand = true, has_int_operand = false, hierarchy = [1], int_operand = 0 : si64, op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.2-scalar_div-58", scope_symbol_id = 449 : i64} : (tensor<1x64x64x512xf16>) -> tensor<1x64x64x512xf16>
      %488 = bufferization.to_memref %487 : memref<1x64x64x512xf16>
      okm.return %488 : memref<1x64x64x512xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x64x64x512xf16>) -> memref<1x64x64x512xf16>
    %273 = "okm.plan_memref"() : () -> memref<1x512x64x64xf16>
    %274 = "okm.wrapper_kernel"(%272, %273) ({
      %486 = bufferization.to_tensor %272 : memref<1x64x64x512xf16>
      %487 = "oneflow.transpose"(%486) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.up_blocks.0.resnets.2-scalar_div-58_transpose_output_0", perm = [0 : si32, 3 : si32, 1 : si32, 2 : si32], scope_symbol_id = 449 : i64} : (tensor<1x64x64x512xf16>) -> tensor<1x512x64x64xf16>
      %488 = bufferization.to_memref %487 : memref<1x512x64x64xf16>
      okm.return %488 : memref<1x512x64x64xf16>
    }) : (memref<1x64x64x512xf16>, memref<1x512x64x64xf16>) -> memref<1x512x64x64xf16>
    %275 = "okm.plan_memref"() : () -> memref<1x512x128x128xf16>
    %276 = "okm.wrapper_kernel"(%274, %275) ({
      %486 = bufferization.to_tensor %274 : memref<1x512x64x64xf16>
      %487 = "oneflow.upsample_nearest_2d"(%486) {data_format = "channels_first", device_name = ["@0:0"], device_tag = "cuda", height_scale = 2.000000e+00 : f64, hierarchy = [1], op_name = "vae_post_process.vae.decoder.up_blocks.0.upsamplers.0-upsample_nearest_2d-59", output_size = [], scope_symbol_id = 456 : i64, width_scale = 2.000000e+00 : f64} : (tensor<1x512x64x64xf16>) -> tensor<1x512x128x128xf16>
      %488 = bufferization.to_memref %487 : memref<1x512x128x128xf16>
      okm.return %488 : memref<1x512x128x128xf16>
    }) : (memref<1x512x64x64xf16>, memref<1x512x128x128xf16>) -> memref<1x512x128x128xf16>
    %277 = "okm.plan_memref"() : () -> memref<1x128x128x512xf16>
    %278 = "okm.wrapper_kernel"(%276, %277) ({
      %486 = bufferization.to_tensor %276 : memref<1x512x128x128xf16>
      %487 = "oneflow.transpose"(%486) {device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "vae_post_process.vae.decoder.up_blocks.0.upsamplers.0.conv-conv2d-60_transpose_input_0", perm = [0 : si32, 2 : si32, 3 : si32, 1 : si32], scope_symbol_id = 468 : i64} : (tensor<1x512x128x128xf16>) -> tensor<1x128x128x512xf16>
      %488 = bufferization.to_memref %487 : memref<1x128x128x512xf16>
      okm.return %488 : memref<1x128x128x512xf16>
    }) : (memref<1x512x128x128xf16>, memref<1x128x128x512xf16>) -> memref<1x128x128x512xf16>
    %279 = "okm.plan_memref"() : () -> memref<1x128x128x512xf16>
    %280 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %281 = "okm.wrapper_kernel"(%278, %55, %56, %279, %280) ({
      %486 = bufferization.to_tensor %278 : memref<1x128x128x512xf16>
      %487 = bufferization.to_tensor %55 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %56 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.up_blocks.0.upsamplers.0.conv-conv2d-60", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 468 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x128x128x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x128x128x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x128x128x512xf16>
      okm.return %490 : memref<1x128x128x512xf16>
    }) : (memref<1x128x128x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x128x128x512xf16>, memref<134217728xi8>) -> memref<1x128x128x512xf16>
    %282 = "okm.plan_memref"() : () -> memref<1x128x128x512xf16>
    %283 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %284 = "okm.plan_memref"() : () -> memref<1x32xf32>
    %285:3 = "okm.wrapper_kernel"(%281, %57, %58, %282, %283, %284) ({
      %486 = bufferization.to_tensor %281 : memref<1x128x128x512xf16>
      %487 = bufferization.to_tensor %57 : memref<512xf16>
      %488 = bufferization.to_tensor %58 : memref<512xf16>
      %y, %mean, %inv_variance = "oneflow.group_norm"(%486, %487, %488) {activation = "silu", affine = true, data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", epsilon = 9.9999999999999995E-7 : f64, hierarchy = [1], num_groups = 32 : si32, op_name = "vae_post_process.vae.decoder.up_blocks.1.resnets.0.norm1-group_norm-61_with_activation", operand_segment_sizes = dense<1> : vector<3xi32>, scope_symbol_id = 482 : i64} : (tensor<1x128x128x512xf16>, tensor<512xf16>, tensor<512xf16>) -> (tensor<1x128x128x512xf16>, tensor<1x32xf32>, tensor<1x32xf32>)
      %489 = bufferization.to_memref %y : memref<1x128x128x512xf16>
      %490 = bufferization.to_memref %mean : memref<1x32xf32>
      %491 = bufferization.to_memref %inv_variance : memref<1x32xf32>
      okm.return %489, %490, %491 : memref<1x128x128x512xf16>, memref<1x32xf32>, memref<1x32xf32>
    }) : (memref<1x128x128x512xf16>, memref<512xf16>, memref<512xf16>, memref<1x128x128x512xf16>, memref<1x32xf32>, memref<1x32xf32>) -> (memref<1x128x128x512xf16>, memref<1x32xf32>, memref<1x32xf32>)
    %286 = "okm.plan_memref"() : () -> memref<1x128x128x512xf16>
    %287 = "okm.plan_memref"() : () -> memref<134217728xi8>
    %288 = "okm.wrapper_kernel"(%285#0, %59, %60, %286, %287) ({
      %486 = bufferization.to_tensor %285#0 : memref<1x128x128x512xf16>
      %487 = bufferization.to_tensor %59 : memref<512x3x3x512xf16>
      %488 = bufferization.to_tensor %60 : memref<512xf16>
      %489 = "oneflow.conv2d"(%486, %487, %488) {data_format = "channels_last", device_name = ["@0:0"], device_tag = "cuda", dilation_rate = [1 : si32, 1 : si32], filters = 512 : si32, groups = 1 : si32, hierarchy = [1], kernel_size = [3 : si32, 3 : si32], op_name = "vae_post_process.vae.decoder.up_blocks.1.resnets.0.conv1-conv2d-63", operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>, padding_before = [1 : si32, 1 : si32], scope_symbol_id = 498 : i64, strides = [1 : si32, 1 : si32]} : (tensor<1x128x128x512xf16>, tensor<512x3x3x512xf16>, tensor<512xf16>) -> tensor<1x128x128x512xf16>
      %490 = bufferization.to_memref %489 : memref<1x128x128x512xf16>
      okm.return %490 : memref<1x128x128x512xf16>
    }) : (memref<1x128x128x512xf16>, memref<512x3x3x512xf16>, memref<512xf16>, memref<1x128x128x512xf16>, memref<134217728xi8>) -> memref<1x128x128x512xf16>
    return
  }
}
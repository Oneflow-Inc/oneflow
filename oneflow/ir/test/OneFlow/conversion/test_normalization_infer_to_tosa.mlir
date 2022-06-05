// RUN: oneflow-opt -lower-oneflow-to-tosa %s | ireec --iree-input-type=tosa --iree-vm-bytecode-module-output-format=flatbuffer-binary -iree-hal-target-backends=dylib-llvm-aot -iree-mlir-to-vm-bytecode-module -


module {
  oneflow.job @GraphModule_0(
    %x:               tensor<1x256x14x14xf32>,
    %moving_mean:     tensor<256xf32>,
    %moving_variance: tensor<256xf32>,
    %gamma:           tensor<256xf32>,
    %beta:            tensor<256xf32>) -> tensor<1x256x14x14xf32> {

    %y = "oneflow.normalization_infer"(%x, %moving_mean, %moving_variance, %gamma, %beta)
    {axis = 1 : si32, device_name = ["@0:0"], device_tag = "cpu", epsilon = 9.99999974E-6 : f32, hierarchy = [1],
     momentum = 0.899999976 : f32, op_name = "model.layer3.5.bn2-normalization-134", operand_segment_sizes = dense<[1, 1, 1, 1, 1, 0]>
      : vector<6xi32>, output_lbns = ["model.layer3.5.bn2-normalization-134/y_0"], result_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>, scope_symbol_id = 4611686018430074879 : i64,
       training = false} :
       (tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<1x256x14x14xf32>
    oneflow.return %y: tensor<1x256x14x14xf32>
  }
}

// RUN: oneflow-opt -lower-oneflow-to-tosa %s | ireec --iree-input-type=tosa --iree-vm-bytecode-module-output-format=flatbuffer-binary -iree-hal-target-backends=dylib-llvm-aot -iree-mlir-to-vm-bytecode-module -

module {
  oneflow.job @GraphModule_1(%arg0: tensor<1x2048xf32>, %arg1: tensor<1000x2048xf32>) ->tensor<1x1000xf32> {
    %0 = "oneflow.matmul"(%arg0, %arg1)
    {alpha = 1.000000e+00 : f64, device_name = ["@0:0"], device_tag = "cpu", hierarchy = [1],
     op_name = "fw.model.fc-matmul-227", output_lbns = ["fw.model.fc-matmul-227/out_0"], scope_symbol_id = 4611686018431234047 : i64,
      transpose_a = false, transpose_b = true} : (tensor<1x2048xf32>, tensor<1000x2048xf32>) -> tensor<1x1000xf32>
    oneflow.return %0:tensor<1x1000xf32>
  }
}

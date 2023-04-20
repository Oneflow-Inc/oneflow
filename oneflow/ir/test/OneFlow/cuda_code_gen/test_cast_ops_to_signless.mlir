// RUN: oneflow-opt %s -cast-ofops-to-signless  | FileCheck %s
// CHECK: unrealized_conversion_cast
func.func @Cast_289__FUSE__ScalarMulByTensor_290() -> tensor<512x2048x1x1xf32> {
    %output_299 = "oneflow.variable"() {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "cuda", hierarchy = [1], op_name = "resnet.layer4.2.conv1.weight", output_lbns = ["resnet.layer4.2.conv1.weight/out"], parallel = #sbp.parallel<[] -> [[#sbp.B]]>, scope_symbol_id = 1995 : i64, shape = [512 : si64, 2048 : si64, 1 : si64, 1 : si64]} : () -> tensor<512x2048x1x1xsi64>
    %0 = "oneflow.cast"(%output_299) {device_name = ["0:0"], device_tag = "cpu", dtype = 2 : i32, hierarchy = [1], op_name = "Cast_1", op_type_name = "cast", scope_symbol_id = 4611686018427416574 : i64} : (tensor<512x2048x1x1xsi64>) -> tensor<512x2048x1x1xf32>
    func.return %0 : tensor<512x2048x1x1xf32>
}
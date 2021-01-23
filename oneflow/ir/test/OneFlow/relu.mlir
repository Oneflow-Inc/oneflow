// RUN: oneflow-opt %s | oneflow-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = "oneflow.constant"() {op_name = "constant1", placement = ["0:0-0"], operand_segment_sizes = dense<[0, 0]> : vector<2xi32>} : () -> tensor<f32>
        %res = "oneflow.relu"(%0) {op_name = "relu1", placement = ["0:0-0"], operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (tensor<f32>) -> tensor<f32>
        // CHECK: %1 = "oneflow.relu"(%0) {op_name = "relu1", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>, placement = ["0:0-0"]} : (tensor<f32>) -> tensor<f32>
        return
    }
}

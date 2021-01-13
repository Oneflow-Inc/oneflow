// RUN: oneflow-opt %s | oneflow-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = "oneflow.constant"() {placement = [], double_value = 2.020000e+03 : f64} : () -> tensor<f32>
        %res = oneflow.relu %0 {op_name = "relu1"} : tensor<f32>
        // CHECK: %{{.*}} = oneflow.relu %{{.*}} : tensor<f32>
        return
    }
}

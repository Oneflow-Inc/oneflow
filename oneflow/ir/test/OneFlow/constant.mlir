// RUN: oneflow-opt %s | oneflow-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %r1 = oneflow.constant {a=11} dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>

        // %r2 = oneflow.constant 1.000000e+00 : tensor<2x3xf64>
        return
    }
}

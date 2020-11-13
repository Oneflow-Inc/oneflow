// RUN: oneflow-opt %s | oneflow-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func @bar() {
        %0 = constant 1 : i32
        // CHECK: %{{.*}} = oneflow.foo %{{.*}} : i32
        %res = oneflow.foo %0 : i32
        return
    }
}

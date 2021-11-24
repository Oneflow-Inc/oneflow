#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_

include "mlir/IR/OpBase.td"

def BnOrderOpInterface : OpInterface<"BnOrder"> {
  let description = [{
    Interface to getting the hard-coded bn
  }];

  let methods = [
    StaticInterfaceMethod<"Get ibn",
                    "void", "inputOrder", (ins), [{
        return ConcreteOp::inputOrderFull();
    }]>,
    StaticInterfaceMethod<"Get obn",
                    "void", "outputOrder", (ins), [{
        return ConcreteOp::outputOrderFull();
    }]>
  ];
}

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_

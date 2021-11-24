#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_

include "mlir/IR/OpBase.td"

def BnOrderOpInterface : OpInterface<"BnOrder"> {
  let description = [{
    Interface to getting the hard-coded bn
  }];

  let methods = [
    StaticInterfaceMethod<"Get ibn",
        "std::vector<std::string>", "inputOrder", (ins), [{
        return ConcreteOp::inputOrder();
    }]>,
    StaticInterfaceMethod<"Get obn",
        "std::vector<std::string>", "outputOrder", (ins), [{
        return ConcreteOp::outputOrder();
    }]>
  ];
}

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_

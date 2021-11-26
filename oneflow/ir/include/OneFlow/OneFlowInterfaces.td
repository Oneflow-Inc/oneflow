#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_

include "mlir/IR/OpBase.td"

def UserOpCompatibleInterface : OpInterface<"UserOpCompatible"> {
  let description = [{
    Interface to getting the hard-coded bn
  }];

  let methods = [
    StaticInterfaceMethod<"",
        "const std::vector<std::string>*", "inputKeys", (ins), [{
        return ConcreteOp::inputKeys();
    }]>,
    StaticInterfaceMethod<"",
        "const std::vector<uint32_t>*", "inputMinimums", (ins), [{
        return ConcreteOp::inputMinimums();
    }]>,
    StaticInterfaceMethod<"",
        "const std::vector<std::string>*", "outputKeys", (ins), [{
        return ConcreteOp::outputKeys();
    }]>,
    StaticInterfaceMethod<"",
        "const std::vector<uint32_t>*", "outputMinimums", (ins), [{
        return ConcreteOp::outputMinimums();
    }]>
  ];
}

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_

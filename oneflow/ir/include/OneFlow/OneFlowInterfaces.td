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
        "const std::vector<std::string>*", "outputKeys", (ins), [{
        return ConcreteOp::outputKeys();
    }]>,
    InterfaceMethod<"",
        "std::pair<unsigned, unsigned>", "getODSOperandIndexAndLength", (ins "unsigned":$index), [{
        return $_op.getODSOperandIndexAndLength(index);
    }]>,
    InterfaceMethod<"",
        "std::pair<unsigned, unsigned>", "getODSResultIndexAndLength", (ins "unsigned":$index), [{
        return $_op.getODSResultIndexAndLength(index);
    }]>
  ];
}

def ControlEdgeCompatibleInterface : OpInterface<"ControlEdgeCompatible"> {
  let description = [{
    Interface to getting control edges
  }];

  let methods = [
    InterfaceMethod<"",
        "::mlir::OperandRange", "dataInputOperands", (ins)
    >,
    InterfaceMethod<"",
        "::mlir::OperandRange", "ctrlInputOperands", (ins)
    >,
    InterfaceMethod<"",
        "::mlir::ResultRange", "dataOutputResults", (ins)
    >,
    InterfaceMethod<"",
        "::mlir::Value", "ctrlOutputResult", (ins)
    >
  ];
}

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_

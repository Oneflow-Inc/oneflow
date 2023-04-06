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
        static std::vector<std::string> val(mlir::oneflow::support::GetInputKeys(ConcreteOp::getOperationName().split('.').second.str()));
        return &val;
    }]>,
    StaticInterfaceMethod<"",
        "const std::vector<std::string>*", "outputKeys", (ins), [{
        static std::vector<std::string> val(mlir::oneflow::support::GetOutputKeys(ConcreteOp::getOperationName().split('.').second.str()));
        return &val;
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
  let cppNamespace = "::mlir::oneflow";
}

def AlternativeOpTypeNameInterface : OpInterface<"HasAlternativeOpTypeName"> {
  let description = [{
    Interface to getting control edges
  }];

  let methods = [
    StaticInterfaceMethod<"",
        "std::string", "getOriginalOpTypeName", (ins)
    >,
    StaticInterfaceMethod<"",
        "const std::vector<std::string>*", "inputKeys", (ins), [{
        static std::vector<std::string> val(mlir::oneflow::support::GetInputKeys(ConcreteOp::getOriginalOpTypeName()));
        return &val;
    }]>,
    StaticInterfaceMethod<"",
        "const std::vector<std::string>*", "outputKeys", (ins), [{
        static std::vector<std::string> val(mlir::oneflow::support::GetOutputKeys(ConcreteOp::getOriginalOpTypeName()));
        return &val;
    }]>,
  ];
  let cppNamespace = "::mlir::oneflow";
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
  let cppNamespace = "::mlir::oneflow";
}

def NoGrad : OpInterface<"NoGrad"> {
  let description = [{
  }];
  let cppNamespace = "::mlir::oneflow";
}

def SupportNonContiguous : OpInterface<"SupportNonContiguous"> {
  let description = [{
  }];
  let cppNamespace = "::mlir::oneflow";
}

def CpuOnly : OpInterface<"CpuOnly"> {
  let description = [{
  }];
  let cppNamespace = "::mlir::oneflow";
}

def NCHWCompatibleInterface : OpInterface<"NCHWCompatible"> {
  let description = [{
    Interface of NCHW compatibility
  }];

  let methods = [
    InterfaceMethod<"",
        "bool", "IsNCHW", (ins)
    >,
    InterfaceMethod<"Create NHWC op and return the new op's results to be transposed",
        "llvm::SmallVector<mlir::Value, 4>", "NchwToNhwc", (ins "llvm::SmallVector<mlir::Value, 4>": $transposed_inputs, "PatternRewriter&": $rewriter)
    >,
    InterfaceMethod<"",
        "llvm::DenseSet<mlir::Value>", "OperandsToTranspose", (ins)
    >,
    InterfaceMethod<"",
        "llvm::DenseSet<mlir::Value>", "ResultsToTranspose", (ins)
    >,
  ];
  let cppNamespace = "::mlir::oneflow";
}

def BiasAddCompatibleInterface : OpInterface<"BiasAddCompatible"> {
  let description = [{
    Interface of ops used as bias add
  }];

  let methods = [
    InterfaceMethod<"",
        "bool", "isLastDim", (ins)
    >,
    InterfaceMethod<"",
        "mlir::Value", "biasAddGetBias", (ins)
    >,
    InterfaceMethod<"",
        "mlir::Value", "biasAddGetOut", (ins)
    >,
  ];
  let cppNamespace = "::mlir::oneflow";
}

def MatMulCompatibleInterface : OpInterface<"MatMulCompatible"> {
  let description = [{
    Interface of ops used as matmul
  }];

  let methods = [
    InterfaceMethod<"is this a transpose_a=false, transpose_b=true matmul",
        "bool", "isLinear", (ins)
    >,
    InterfaceMethod<"",
        "mlir::Value", "matMulGetX", (ins)
    >,
    InterfaceMethod<"",
        "mlir::Value", "matMulGetW", (ins)
    >,
    InterfaceMethod<"",
        "mlir::Value", "matMulGetY", (ins)
    >,
  ];
  let cppNamespace = "::mlir::oneflow";
}


#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWINTERFACES_H_

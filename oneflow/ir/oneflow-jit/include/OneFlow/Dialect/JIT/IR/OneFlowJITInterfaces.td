#ifndef ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_DIALECT_JIT_IR_JITINTERFACES_H_
#define ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_DIALECT_JIT_IR_JITINTERFACES_H_

include "mlir/IR/OpBase.td"

def EagerMaterializeInterface : OpInterface<"EagerMaterialize"> {
  let cppNamespace = "::mlir::oneflow::jit";
  let methods = [];

  let extraClassDeclaration = [{
    constexpr const static ::llvm::StringLiteral
      kEagerTensor = "oneflow.eager";
  }];
}


#endif  // ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_DIALECT_JIT_IR_JITINTERFACES_H_

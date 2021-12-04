#ifndef ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_DIALECT_JIT_IR_JITBASE_H_
#define ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_DIALECT_JIT_IR_JITBASE_H_

include "mlir/IR/OpBase.td"

def OneFlowJIT_Dialect : Dialect {
  let name = "oneflow_jit";
  let cppNamespace = "::mlir::oneflow::jit";
}

#endif  // ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_DIALECT_JIT_IR_JITBASE_H_

#ifndef ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_DIALECT_JIT_IR_JITOPS_H_
#define ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_DIALECT_JIT_IR_JITOPS_H_

include "OneFlowJITBase.td"
include "OneFlowJITInterfaces.td"

def OneFlow_EagerMaterializeOp : Op<OneFlowJIT_Dialect, "eager_materialize", [DeclareOpInterfaceMethods<EagerMaterializeInterface>]> {
  let arguments = (ins
    AnyType:$lazy_tensor,
    AnyType:$eager_tensor
  );
}

#endif  // ONEFLOW_IR_ONEFLOW_JIT_INCLUDE_ONEFLOW_DIALECT_JIT_IR_JITOPS_H_

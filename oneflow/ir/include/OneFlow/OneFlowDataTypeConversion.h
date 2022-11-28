#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWDATATYPECONVERSION_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWDATATYPECONVERSION_H_

#include "mlir/IR/Builders.h"
#include "OneFlow/OneFlowSupport.h"

namespace mlir {

namespace oneflow {

Type getTypeFromOneFlowDataType(MLIRContext* context, ::oneflow::DataType dt);
llvm::Optional<Type> getTypeFromOneFlowDataType(Builder& builder, ::oneflow::DataType dt);

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWDATATYPECONVERSION_H_

#ifndef ONEFLOW_ONEFLOWOPS_H
#define ONEFLOW_ONEFLOWOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "OneFlow/OneFlowEnums.h.inc"

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.h.inc"

#endif  // ONEFLOW_ONEFLOWOPS_H

/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKM_OKMOPS_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKM_OKMOPS_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "OneFlow/OKM/OKMAttributes.h"

namespace mlir {
namespace func {
class FuncOp;
}  // namespace func
}  // namespace mlir

#define GET_OP_CLASSES
#include "OneFlow/OKMOps.h.inc"

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKM_OKMOPS_H_

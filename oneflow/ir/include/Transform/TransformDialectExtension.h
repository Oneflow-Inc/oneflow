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
#ifndef ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_H_
#define ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_H_

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/MatchInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class DialectRegistry;

namespace oneflow {
namespace transform_dialect {
/// Registers the test extension to the Transform dialect.
void registerTransformDialectExtension(::mlir::DialectRegistry& registry);
void registerTransformDialectEraseSchedulePass();
void registerTransformDialectInterpreterPass();

struct ApplyPatternsOpPatterns {
  bool canonicalization = false;
  bool cse = false;
};

}  // namespace transform_dialect

}  // namespace oneflow
}  // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "Transform/TransformDialectExtensionTypes.h.inc"

#define GET_OP_CLASSES
#include "Transform/TransformDialectExtension.h.inc"

#endif  // ONEFLOW_IR_INCLUDE_TRANSOFRM_TRANSFORM_DIALECT_EXTENSION_H_

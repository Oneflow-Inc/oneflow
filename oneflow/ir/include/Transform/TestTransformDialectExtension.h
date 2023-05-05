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
//===- TestTransformDialectExtension.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an extension of the MLIR Transform dialect for testing
// purposes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TESTTRANSFORMDIALECTEXTENSION_H
#define MLIR_TESTTRANSFORMDIALECTEXTENSION_H

#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/MatchInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
class DialectRegistry;

namespace transform {
/// Registers the test extension to the Transform dialect.
void registerTestTransformDialectExtension(::mlir::DialectRegistry& registry);
void registerTestTransformDialectEraseSchedulePass();
void registerTestTransformDialectInterpreterPass();
}  // namespace transform

}  // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "Transform/TestTransformDialectExtensionTypes.h.inc"

#define GET_OP_CLASSES
#include "Transform/TestTransformDialectExtension.h.inc"

#endif  // MLIR_TESTTRANSFORMDIALECTEXTENSION_H

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
//===- TestTransformStateExtension.h - Test Utility -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an TransformState extension for the purpose of testing the
// relevant APIs.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_LIB_DIALECT_TRANSFORM_TESTTRANSFORMSTATEEXTENSION_H
#define MLIR_TEST_LIB_DIALECT_TRANSFORM_TESTTRANSFORMSTATEEXTENSION_H

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

namespace mlir {
namespace test {
class TestTransformStateExtension : public transform::TransformState::Extension {
 public:
  TestTransformStateExtension(transform::TransformState& state, StringAttr message)
      : Extension(state), message(message) {}

  StringRef getMessage() const { return message.getValue(); }

  LogicalResult updateMapping(Operation* previous, Operation* updated);

 private:
  StringAttr message;
};
}  // namespace test
}  // namespace mlir

#endif  // MLIR_TEST_LIB_DIALECT_TRANSFORM_TESTTRANSFORMSTATEEXTENSION_H

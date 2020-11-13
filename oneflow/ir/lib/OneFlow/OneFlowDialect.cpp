//===- OneFlowDialect.cpp - OneFlow dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"

using namespace mlir;
using namespace mlir::oneflow;

//===----------------------------------------------------------------------===//
// OneFlow dialect.
//===----------------------------------------------------------------------===//

void OneFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "OneFlow/OneFlowOps.cpp.inc"
      >();
}

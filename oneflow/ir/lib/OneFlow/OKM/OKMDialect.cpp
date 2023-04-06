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
#include "OneFlow/OKM/OKMDialect.h"
#include "OneFlow/OKM/OKMOps.h"
#include "OneFlow/OKM/OKMAttributes.h"

#include "OneFlow/OKM/passes.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "OneFlow/OKMDialect.cpp.inc"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"

#define GET_ATTRDEF_CLASSES
#include "OneFlow/OKMAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "OneFlow/OKMOps.cpp.inc"

namespace mlir {

namespace okm {

void OKMDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "OneFlow/OKMOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "OneFlow/OKMAttributes.cpp.inc"
      >();
}

void registerAllPasses() {
  registerExtractOKMTensorPassPass();
  registerWrapOKMKernelPassPass();
  registerOptOKMMemrefPassPass();
  registerConvertOKMToOKLPassPass();
}

}  // namespace okm

}  // namespace mlir

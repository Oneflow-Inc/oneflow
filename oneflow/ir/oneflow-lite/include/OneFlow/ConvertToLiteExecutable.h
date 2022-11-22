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
#ifndef ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_CONVERTTOLITEEXECUTABLE_H_
#define ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_CONVERTTOLITEEXECUTABLE_H_

#include "llvm/ADT/SmallString.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "OneFlow/FlatbufferUtils.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include "schemas/executable_generated.h"
#pragma GCC diagnostic pop

namespace mlir {
namespace oneflow {

namespace lite {

class LiteExecutable {
 public:
  LiteExecutable() = default;

 private:
  FlatbufferBuilder builder_;
};

typedef struct ConvertOptions {
  llvm::SmallString<128> target;
} ConvertOptions;

LogicalResult ConvertToLiteExecutable(MLIRContext* context, ModuleOp module, ConvertOptions options,
                                      LiteExecutable* executable);

}  // namespace lite

}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_CONVERTTOLITEEXECUTABLE_H_

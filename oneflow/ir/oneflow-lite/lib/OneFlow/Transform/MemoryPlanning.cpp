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
#include "OneFlow/Transform/MemoryPlanning.h"

namespace mlir {
namespace oneflow {
namespace lite {

struct MemoryPlanningPass : public PassWrapper<MemoryPlanningPass, OperationPass<>> {
  void runOnOperation() override {
    // TODO
  }
};

std::unique_ptr<mlir::Pass> createLiteMemoryPlanningPass() {
  return std::unique_ptr<mlir::Pass>(new MemoryPlanningPass);
}

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir

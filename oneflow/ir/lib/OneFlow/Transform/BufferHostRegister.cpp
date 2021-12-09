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
#include <iostream>
#include <string>
#include "OneFlow/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

class BufferHostRegisterPass : public BufferHostRegisterPassBase<BufferHostRegisterPass> {
  void runOnFunction() override {
    getFunction()->walk([&](memref::AllocOp alloc) {
      auto ranked_type = alloc.getResult().getType().cast<MemRefType>();
      Type unranked_type =
          UnrankedMemRefType::get(ranked_type.getElementType(), ranked_type.getMemorySpace());
      OpBuilder builder(alloc);
      builder.setInsertionPointAfter(alloc);
      Value casted = builder.create<memref::CastOp>(alloc->getLoc(), unranked_type, alloc);
      builder.create<gpu::HostRegisterOp>(alloc->getLoc(), casted);
    });
  }
};

}  // namespace

namespace mlir {

namespace oneflow {

std::unique_ptr<Pass> createBufferHostRegisterPass() {
  return std::make_unique<BufferHostRegisterPass>();
}

}  // namespace oneflow

}  // namespace mlir

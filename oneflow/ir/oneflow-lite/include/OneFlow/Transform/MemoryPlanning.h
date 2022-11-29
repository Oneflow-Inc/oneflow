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
#ifndef ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_TRANSFORM_MEMORYPLANNING_H_
#define ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_TRANSFORM_MEMORYPLANNING_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace oneflow {
namespace lite {

struct LiteBufferSegment {
  StringRef device;
  size_t size;
  size_t alignment;
};

class LiteBufferStrategy {
 public:
  LiteBufferStrategy() = default;

  const llvm::SmallVector<LiteBufferSegment, 4>& getSegments() const { return segments; }

  llvm::SmallVector<LiteBufferSegment, 4>& getSegments() { return segments; }

  int getValueSegmentId(Value value) const;
  size_t getValueSegmentOffset(Value value) const;

  LogicalResult insertValue(Value value, int segmentId, size_t segmentOffset);

 private:
  llvm::SmallVector<LiteBufferSegment, 4> segments;
  struct ValueSegmentInfo {
    int segmentId;
    size_t segmentOffset;
  };
  llvm::DenseMap<Value, ValueSegmentInfo> valueSegmentInfos;
};

std::unique_ptr<mlir::Pass> createLiteMemoryPlanningPass(LiteBufferStrategy* strategy);

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_ONEFLOW_LITE_INCLUDE_ONEFLOW_TRANSFORM_MEMORYPLANNING_H_

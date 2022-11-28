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

#include <assert.h>

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowOpTraits.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace oneflow {
namespace lite {

static llvm::DenseMap<StringAttr, size_t> deviceSegmentSize;
static llvm::DenseMap<Value, size_t> valueSegmentOffset;

const llvm::DenseMap<StringAttr, size_t>& getDeviceSegmentSize() { return deviceSegmentSize; }
const llvm::DenseMap<Value, size_t>& getValueSegmentOffset() { return valueSegmentOffset; }

class ValueLiveness {
 public:
  ValueLiveness() = default;

  void addValue(Value value, size_t liveStart, size_t liveEnd) {
    liveness[value] = LiveRange{liveStart, liveEnd};
  }

  bool isLivenessOverlap(Value lhs, Value rhs) {
    LiveRange lhs_liveness = liveness[lhs];
    LiveRange rhs_liveness = liveness[rhs];
    return lhs_liveness.liveEnd < rhs_liveness.liveStart
           || lhs_liveness.liveStart > rhs_liveness.liveEnd;
  }

 private:
  struct LiveRange {
    size_t liveStart;
    size_t liveEnd;
  };
  llvm::DenseMap<Value, LiveRange> liveness;
};

struct MemoryPlanningPass : public PassWrapper<MemoryPlanningPass, OperationPass<ModuleOp>> {
  ValueLiveness valueLiveness;
  llvm::SmallVector<Value, 4> sortedValues;

  void runOnOperation() override {
    computeValueLiveness();
    computeValueSizeAndSort();
  }

  void computeValueLiveness();
  void computeValueSizeAndSort();
  void doMemoryPlanning();
  bool canShareMemoryBlock(Value value, llvm::SmallVector<Value, 4> block);
};

void MemoryPlanningPass::computeValueLiveness() {
  llvm::SmallVector<Operation*, 4> opList;
  llvm::DenseMap<Operation*, size_t> opOrdering;
  llvm::DenseMap<Value, size_t> liveEnds;

  // Compute value liveness
  getOperation().walk([&](Operation* op) {
    if (!op->hasTrait<OpTrait::IsOpConfCompatible>() || llvm::dyn_cast<OutputOp>(op)) { return; }
    opOrdering[op] = opOrdering.size();
    opList.push_back(op);
  });
  for (Operation* op : llvm::reverse(opList)) {
    size_t ordering = opOrdering[op];
    for (Value operand : op->getOperands()) {
      if (liveEnds.find(operand) == liveEnds.end()) { liveEnds[operand] = ordering; }
    }
    for (Value result : op->getResults()) {
      size_t liveEnd = opOrdering.size();
      const auto& it = liveEnds.find(result);
      if (it != liveEnds.end()) { liveEnd = it->second; }
      valueLiveness.addValue(result, ordering, liveEnd);
    }
  }
}

static bool isDynamicTensorType(TensorType value) {
  for (auto dim : value.getShape()) {
    if (dim == -1) { return true; }
  }
  return false;
}

/// Returns the bitwidth of a scalar or vector type.
static size_t getTensorBitSize(TensorType value) {
  auto type = value.getElementType();
  assert(type.isIntOrFloat());
  if (isDynamicTensorType(value)) { return 0; }
  int64_t num = 1;
  for (auto dim : value.getShape()) { num *= dim; }
  return num * type.getIntOrFloatBitWidth();
}

void MemoryPlanningPass::computeValueSizeAndSort() {
  llvm::SetVector<Value, llvm::SmallVector<Value, 4>> valueList;
  getOperation().walk([&](Operation* op) {
    if (!op->hasTrait<OpTrait::IsOpConfCompatible>() || llvm::dyn_cast<InputOp>(op)
        || llvm::dyn_cast<OutputOp>(op)) {
      return;
    }
    valueList.insert(op->getOperands().begin(), op->getOperands().end());
    valueList.insert(op->getResults().begin(), op->getResults().end());
  });
  sortedValues = valueList.takeVector();
  llvm::sort(sortedValues.begin(), sortedValues.end(), [](Value lhs, Value rhs) {
    assert(lhs.getType().isa<TensorType>());
    assert(rhs.getType().isa<TensorType>());
    return getTensorBitSize(lhs.getType().cast<TensorType>())
           > getTensorBitSize(rhs.getType().cast<TensorType>());
  });
}

bool MemoryPlanningPass::canShareMemoryBlock(Value value, llvm::SmallVector<Value, 4> block) {
  if (isDynamicTensorType(value.getType().cast<TensorType>())) { return false; }
  auto device = value.getDefiningOp()->getAttrOfType<StringAttr>(
      OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr());
  for (auto v : block) {
    if (device
        != v.getDefiningOp()->getAttrOfType<StringAttr>(
            OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr())) {
      return false;
    }
    if (valueLiveness.isLivenessOverlap(value, v)) { return false; }
  }
  return true;
}

void MemoryPlanningPass::doMemoryPlanning() {
  if (sortedValues.empty()) { return; }
  llvm::SmallVector<llvm::SmallVector<Value, 4>, 4> blocks;
  for (auto value : sortedValues) {
    bool newBlock = true;
    for (auto& block : blocks) {
      if (canShareMemoryBlock(value, block)) {
        block.push_back(value);
        newBlock = false;
      }
    }
    if (newBlock) { blocks.push_back(llvm::SmallVector<Value, 4>{value}); }
  }

  for (auto& block : blocks) {
    auto device = block.front().getDefiningOp()->getAttrOfType<StringAttr>(
        OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr());
    size_t offset = 0;
    auto it = deviceSegmentSize.find(device);
    if (it != deviceSegmentSize.end()) { offset = it->second; }
    size_t block_size = 0;
    for (auto value : block) {
      size_t bitsize = getTensorBitSize(value.getType().cast<TensorType>());
      if (bitsize > block_size) { block_size = bitsize; }
      valueSegmentOffset[value] = offset;
    }
    block_size = (block_size + 7) / 8;      // bytesize
    block_size = (block_size + 512) / 512;  // align as 512 bytes
    deviceSegmentSize[device] = offset + block_size;
  }
}

std::unique_ptr<mlir::Pass> createLiteMemoryPlanningPass() {
  return std::unique_ptr<mlir::Pass>(new MemoryPlanningPass);
}

}  // namespace lite
}  // namespace oneflow
}  // namespace mlir
